using System;
using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen35;

public class Qwen35RMSNorm : nn.Module
{
    private readonly double _eps;

    public Qwen35RMSNorm(long hiddenSize, double eps = 1e-6, string name = "norm") : base(name)
    {
        _eps = eps;
        register_parameter("weight", Parameter(torch.zeros(hiddenSize)));
    }

    public Tensor forward(Tensor hiddenStates)
    {
        using var scope = torch.NewDisposeScope();

        var inputDtype = hiddenStates.dtype;
        var hiddenStatesFp32 = hiddenStates.to(ScalarType.Float32);
        var variance = hiddenStatesFp32.pow(2).mean(new long[] { -1 }, keepdim: true);
        var normalized = hiddenStatesFp32 * torch.rsqrt(variance + _eps);

        var weight = get_parameter("weight");
        normalized = normalized * (1.0 + weight.to(ScalarType.Float32));

        return scope.MoveToOuter(normalized.to(inputDtype));
    }
}

public class Qwen35RMSNormGated : nn.Module
{
    private readonly Parameter _weight;
    private readonly double _eps;

    public Qwen35RMSNormGated(long hiddenSize, double eps = 1e-6, string name = "norm") : base(name)
    {
        _weight = Parameter(torch.ones(hiddenSize));
        _eps = eps;
        register_parameter("weight", _weight);
    }

    public Tensor forward(Tensor hiddenStates, Tensor gate)
    {
        using var scope = torch.NewDisposeScope();
        var inputDtype = hiddenStates.dtype;
        
        var hsFloat = hiddenStates.to(ScalarType.Float32);
        var variance = hsFloat.pow(2).mean(new long[] { -1 }, keepdim: true);
        
        hsFloat = hsFloat * torch.rsqrt(variance + _eps);
        
        var weight = get_parameter("weight");
        hsFloat = weight * hsFloat.to(inputDtype);
        
        hsFloat = hsFloat * torch.nn.functional.silu(gate.to(ScalarType.Float32));
        
        return scope.MoveToOuter(hsFloat.to(inputDtype));
    }
}

public class Qwen35GatedDeltaNet : nn.Module
{
    private readonly long _hiddenSize;
    private readonly int _numVHeads;
    private readonly int _numKHeads;
    private readonly int _headKDim;
    private readonly int _headVDim;
    private readonly int _keyDim;
    private readonly int _valueDim;
    private readonly int _convKernelSize;
    private readonly int _layerIdx;

    private readonly int _convDim;
    private readonly Conv1d _conv1d;

    private readonly Parameter _dtBias;
    private readonly Parameter _aLog;

    private readonly Qwen35RMSNormGated _norm;
    private readonly Linear _outProj;

    private readonly Linear _inProjQkv;
    private readonly Linear _inProjZ;
    private readonly Linear _inProjB;
    private readonly Linear _inProjA;

    public Qwen35GatedDeltaNet(Qwen35Config config, int layerIdx, string name = "gated_delta_net") : base(name)
    {
        _hiddenSize = config.HiddenSize;
        _numVHeads = config.LinearNumValueHeads;
        _numKHeads = config.LinearNumKeyHeads;
        _headKDim = config.LinearKeyHeadDim;
        _headVDim = config.LinearValueHeadDim;
        _keyDim = _headKDim * _numKHeads;
        _valueDim = _headVDim * _numVHeads;
        _convKernelSize = config.LinearConvKernelDim;
        _layerIdx = layerIdx;

        _convDim = _keyDim * 2 + _valueDim;
        _conv1d = Conv1d(_convDim, _convDim, _convKernelSize, padding: _convKernelSize - 1, groups: _convDim, bias: false);
        
        _dtBias = Parameter(torch.ones(_numVHeads));
        var aTensor = torch.empty(_numVHeads).uniform_(0, 16);
        _aLog = Parameter(torch.log(aTensor));

        _norm = new Qwen35RMSNormGated(_headVDim, config.RmsNormEps, name: "norm");
        _outProj = Linear(_valueDim, _hiddenSize, hasBias: false);

        _inProjQkv = Linear(_hiddenSize, _keyDim * 2 + _valueDim, hasBias: false);
        _inProjZ = Linear(_hiddenSize, _valueDim, hasBias: false);
        _inProjB = Linear(_hiddenSize, _numVHeads, hasBias: false);
        _inProjA = Linear(_hiddenSize, _numVHeads, hasBias: false);

        register_module("conv1d", _conv1d);
        register_module("norm", _norm);
        register_module("out_proj", _outProj);
        register_module("in_proj_qkv", _inProjQkv);
        register_module("in_proj_z", _inProjZ);
        register_module("in_proj_b", _inProjB);
        register_module("in_proj_a", _inProjA);
        register_parameter("dt_bias", _dtBias);
        register_parameter("A_log", _aLog);
    }

    public Tensor forward(Tensor hiddenStates, KVCache? kvCache = null, Tensor? attentionMask = null)
    {
        using var scope = torch.NewDisposeScope();

        var bsz = hiddenStates.shape[0];
        var seqLen = hiddenStates.shape[1];
        var usePrecomputedStates = kvCache is not null && kvCache.HasLinearState(_layerIdx) && seqLen == 1;

        var mixedQkv = _inProjQkv.forward(hiddenStates).transpose(1, 2);
        var z = _inProjZ.forward(hiddenStates).reshape(bsz, seqLen, -1, _headVDim);
        var b = _inProjB.forward(hiddenStates);
        var a = _inProjA.forward(hiddenStates);

        if (usePrecomputedStates)
        {
            var convState = kvCache!.GetConvState(_layerIdx)
                ?? throw new InvalidOperationException($"Missing linear attention conv state for layer {_layerIdx}");
            mixedQkv = GatedDeltaNetHelper.TorchCausalConv1dUpdate(
                mixedQkv,
                convState,
                _conv1d.weight.squeeze(1),
                _conv1d.bias);
        }
        else
        {
            if (kvCache is not null)
            {
                var padLen = Math.Max(_convKernelSize - (int)mixedQkv.shape[2], 0);
                var convState = torch.nn.functional.pad(mixedQkv, new long[] { padLen, 0 });
                scope.Detach(convState);
                kvCache.SetConvState(_layerIdx, scope.MoveToOuter(convState));
            }

            mixedQkv = torch.nn.functional.silu(_conv1d.forward(mixedQkv)[.., .., ..(int)seqLen]);
        }

        mixedQkv = mixedQkv.transpose(1, 2);

        var splitSizes = new long[] { _keyDim, _keyDim, _valueDim };
        var qkvChunks = mixedQkv.split(splitSizes, dim: -1);
        var query = qkvChunks[0].reshape(bsz, seqLen, -1, _headKDim);
        var key = qkvChunks[1].reshape(bsz, seqLen, -1, _headKDim);
        var value = qkvChunks[2].reshape(bsz, seqLen, -1, _headVDim);

        var dtBias = get_parameter("dt_bias");
        var aLog = get_parameter("A_log");

        var beta = b.sigmoid();
        var g = -aLog.to(ScalarType.Float32).exp() * torch.nn.functional.softplus(a.to(ScalarType.Float32) + dtBias);

        if (_numVHeads / _numKHeads > 1)
        {
            query = query.repeat_interleave(_numVHeads / _numKHeads, dim: 2);
            key = key.repeat_interleave(_numVHeads / _numKHeads, dim: 2);
        }

        var initialState = usePrecomputedStates ? kvCache!.GetRecurrentState(_layerIdx) : null;
        var (coreAttnOut, lastRecurrentState) = GatedDeltaNetHelper.TorchRecurrentGatedDeltaRule(
            query,
            key,
            value,
            g,
            beta,
            initialState: initialState,
            outputFinalState: kvCache is not null,
            useQkL2NormInKernel: true);

        if (kvCache is not null)
        {
            if (lastRecurrentState is not null)
            {
                scope.Detach(lastRecurrentState);
                kvCache.SetRecurrentState(_layerIdx, scope.MoveToOuter(lastRecurrentState));
            }
        }

        coreAttnOut = coreAttnOut.reshape(-1, _headVDim);
        var zFlat = z.reshape(-1, _headVDim);
        
        coreAttnOut = _norm.forward(coreAttnOut, zFlat);
        coreAttnOut = coreAttnOut.reshape(bsz, seqLen, -1);

        var output = _outProj.forward(coreAttnOut);

        return scope.MoveToOuter(output);
    }
}
