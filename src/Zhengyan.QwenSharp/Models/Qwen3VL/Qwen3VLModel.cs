using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Qwen2VL;
using Zhengyan.QwenSharp.Models.Common;
using Zhengyan.QwenSharp.Models.Vision;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen3VL;

public class Qwen3VLVisionPatchEmbed : nn.Module
{
    private readonly int _patchSize;
    private readonly int _temporalPatchSize;
    private readonly int _inChannels;
    private readonly int _embedDim;
    private readonly Conv3d _proj;

    public Qwen3VLVisionPatchEmbed(Qwen3VLVisionConfig config, string name = "patch_embed") : base(name)
    {
        _patchSize = config.PatchSize;
        _temporalPatchSize = config.TemporalPatchSize;
        _inChannels = config.InChannels;
        _embedDim = config.HiddenSize;

        _proj = Conv3d(_inChannels, _embedDim, kernel_size: (_temporalPatchSize, _patchSize, _patchSize), stride: (_temporalPatchSize, _patchSize, _patchSize), bias: true);
        register_module("proj", _proj);
    }

    public Tensor forward(Tensor hiddenStates)
    {
        using var scope = torch.NewDisposeScope();
        var targetDtype = _proj.weight.dtype;
        hiddenStates = hiddenStates.view(-1, _inChannels, _temporalPatchSize, _patchSize, _patchSize);
        hiddenStates = _proj.forward(hiddenStates.to_type(targetDtype)).view(-1, _embedDim);
        return scope.MoveToOuter(hiddenStates);
    }
}

public class Qwen3VLMerger : nn.Module
{
    private readonly LayerNorm _norm;
    private readonly Linear _linearFc1;
    private readonly Linear _linearFc2;
    private readonly int _hiddenSize;
    private readonly bool _normalizeMergedDim;

    public Qwen3VLMerger(int outHiddenSize, int contextDim, int spatialMergeSize, string name, bool normalizeMergedDim = false) : base(name)
    {
        _hiddenSize = contextDim * spatialMergeSize * spatialMergeSize;
        _normalizeMergedDim = normalizeMergedDim;
        _norm = LayerNorm(new long[] { _normalizeMergedDim ? _hiddenSize : contextDim }, eps: 1e-6);
        _linearFc1 = Linear(_hiddenSize, _hiddenSize);
        _linearFc2 = Linear(_hiddenSize, outHiddenSize);

        register_module("norm", _norm);
        register_module("linear_fc1", _linearFc1);
        register_module("linear_fc2", _linearFc2);
    }

    public Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        x = _normalizeMergedDim
            ? _norm.forward(x.view(-1, _hiddenSize))
            : _norm.forward(x).view(-1, _hiddenSize);
        x = _linearFc2.forward(torch.nn.functional.gelu(_linearFc1.forward(x)));
        return scope.MoveToOuter(x);
    }
}

public class Qwen3VLVisionAttention : nn.Module
{
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly Linear _qkv;
    private readonly Linear _proj;

    public Qwen3VLVisionAttention(Qwen3VLVisionConfig config, string name = "attn") : base(name)
    {
        _dim = config.HiddenSize;
        _numHeads = config.NumHeads;
        _headDim = _dim / _numHeads;

        _qkv = Linear(_dim, _dim * 3, hasBias: true);
        _proj = Linear(_dim, _dim, hasBias: true);

        register_module("qkv", _qkv);
        register_module("proj", _proj);
    }

    private Tensor RotateHalf(Tensor x)
    {
        var half = x.shape[^1] / 2;
        var x1 = x.slice(-1, 0, half, 1);
        var x2 = x.slice(-1, half, x.shape[^1], 1);
        return torch.cat(new[] { -x2, x1 }, dim: -1);
    }

    public Tensor forward(Tensor hiddenStates, Tensor cuSeqlens, Tensor cos, Tensor sin)
    {
        using var scope = torch.NewDisposeScope();

        var seqLength = hiddenStates.shape[0];
        var qkvOutput = _qkv.forward(hiddenStates).reshape(seqLength, 3, _numHeads, _headDim).permute(1, 0, 2, 3);

        var q = qkvOutput[0];
        var k = qkvOutput[1];
        var v = qkvOutput[2];

        var qf = q.to_type(ScalarType.Float32);
        var kf = k.to_type(ScalarType.Float32);
        var cosF = cos.unsqueeze(-2).to_type(ScalarType.Float32);
        var sinF = sin.unsqueeze(-2).to_type(ScalarType.Float32);

        q = ((qf * cosF) + (RotateHalf(qf) * sinF)).to_type(hiddenStates.dtype).transpose(0, 1).unsqueeze(0);
        k = ((kf * cosF) + (RotateHalf(kf) * sinF)).to_type(hiddenStates.dtype).transpose(0, 1).unsqueeze(0);
        v = v.transpose(0, 1).unsqueeze(0);

        var cuSeqlensArray = cuSeqlens.dtype == ScalarType.Int32 ? Array.ConvertAll(cuSeqlens.data<int>().ToArray(), static x => (long)x) : cuSeqlens.data<long>().ToArray();
        var attnOutputParts = new List<Tensor>();

        for (int i = 0; i < cuSeqlensArray.Length - 1; i++)
        {
            var len = cuSeqlensArray[i + 1] - cuSeqlensArray[i];
            var qChunk = q.narrow(2, cuSeqlensArray[i], len);
            var kChunk = k.narrow(2, cuSeqlensArray[i], len);
            var vChunk = v.narrow(2, cuSeqlensArray[i], len);
            attnOutputParts.Add(torch.nn.functional.scaled_dot_product_attention(qChunk, kChunk, vChunk, null, 0.0, false).transpose(1, 2));
        }

        var attnOutput = torch.cat(attnOutputParts, dim: 1).reshape(seqLength, -1).contiguous();
        return scope.MoveToOuter(_proj.forward(attnOutput));
    }
}

public class Qwen3VLVisionMLP : nn.Module
{
    private readonly Linear _linearFc1;
    private readonly Linear _linearFc2;
    private readonly string _hiddenAct;

    public Qwen3VLVisionMLP(Qwen3VLVisionConfig config, string name = "mlp") : base(name)
    {
        _linearFc1 = Linear(config.HiddenSize, config.IntermediateSize, hasBias: true);
        _linearFc2 = Linear(config.IntermediateSize, config.HiddenSize, hasBias: true);
        _hiddenAct = config.HiddenAct;

        register_module("linear_fc1", _linearFc1);
        register_module("linear_fc2", _linearFc2);
    }

    private Tensor ApplyActivation(Tensor x)
    {
        if (string.Equals(_hiddenAct, "gelu_pytorch_tanh", StringComparison.OrdinalIgnoreCase))
        {
            using var cubic = x.pow(3);
            using var inner = x + (0.044715 * cubic);
            using var scaled = inner * Math.Sqrt(2.0 / Math.PI);
            return 0.5 * x * (1.0 + scaled.tanh());
        }

        return torch.nn.functional.gelu(x);
    }

    public Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        x = ApplyActivation(_linearFc1.forward(x));
        x = _linearFc2.forward(x);
        return scope.MoveToOuter(x);
    }
}

public class Qwen3VLVisionBlock : nn.Module
{
    private readonly LayerNorm _norm1;
    private readonly LayerNorm _norm2;
    private readonly Qwen3VLVisionAttention _attn;
    private readonly Qwen3VLVisionMLP _mlp;

    public Qwen3VLVisionBlock(Qwen3VLVisionConfig config, string name = "block") : base(name)
    {
        _norm1 = LayerNorm(new long[] { config.HiddenSize }, eps: 1e-6);
        _norm2 = LayerNorm(new long[] { config.HiddenSize }, eps: 1e-6);
        _attn = new Qwen3VLVisionAttention(config, name: "attn");
        _mlp = new Qwen3VLVisionMLP(config, name: "mlp");

        register_module("norm1", _norm1);
        register_module("norm2", _norm2);
        register_module("attn", _attn);
        register_module("mlp", _mlp);
    }

    public Tensor forward(Tensor hiddenStates, Tensor cuSeqlens, Tensor cos, Tensor sin)
    {
        using var scope = torch.NewDisposeScope();
        var residual = hiddenStates;
        hiddenStates = _norm1.forward(hiddenStates);
        hiddenStates = residual + _attn.forward(hiddenStates, cuSeqlens, cos, sin);

        residual = hiddenStates;
        hiddenStates = _norm2.forward(hiddenStates);
        hiddenStates = residual + _mlp.forward(hiddenStates);
        return scope.MoveToOuter(hiddenStates);
    }
}

public class Qwen3VisionTransformer : nn.Module
{
    private readonly int _spatialMergeSize;
    private readonly int[] _deepstackVisualIndexes;
    private readonly Qwen3VLVisionPatchEmbed _patchEmbed;
    private readonly Embedding _posEmbed;
    private readonly int _numGridPerSide;
    private readonly VisionRotaryEmbedding _rotaryPosEmb;
    private readonly ModuleList<Qwen3VLVisionBlock> _blocks;
    private readonly Qwen3VLMerger _merger;
    private readonly ModuleList<Qwen3VLMerger> _deepstackMergers;

    public Qwen3VisionTransformer(Qwen3VLVisionConfig config, string name = "visual") : base(name)
    {
        _spatialMergeSize = config.SpatialMergeSize;
        _deepstackVisualIndexes = config.DeepstackVisualIndexes ?? [];

        _patchEmbed = new Qwen3VLVisionPatchEmbed(config, name: "patch_embed");
        _posEmbed = Embedding(config.NumPositionEmbeddings, config.HiddenSize);
        _numGridPerSide = (int)Math.Sqrt(config.NumPositionEmbeddings);
        int headDim = config.HiddenSize / config.NumHeads;
        _rotaryPosEmb = new VisionRotaryEmbedding(headDim / 2, name: "rotary_pos_emb");

        var blocksList = new List<Qwen3VLVisionBlock>();
        for (int i = 0; i < config.Depth; i++)
        {
            blocksList.Add(new Qwen3VLVisionBlock(config, name: $"{i}"));
        }
        _blocks = ModuleList(blocksList.ToArray());

        _merger = new Qwen3VLMerger(config.OutHiddenSize, config.HiddenSize, config.SpatialMergeSize, name: "merger");

        var deepstackList = new List<Qwen3VLMerger>();
        for (int i = 0; i < _deepstackVisualIndexes.Length; i++)
        {
            deepstackList.Add(new Qwen3VLMerger(config.OutHiddenSize, config.HiddenSize, config.SpatialMergeSize, name: $"{i}", normalizeMergedDim: true));
        }
        _deepstackMergers = ModuleList(deepstackList.ToArray());

        register_module("patch_embed", _patchEmbed);
        register_module("pos_embed", _posEmbed);
        register_module("rotary_pos_emb", _rotaryPosEmb);
        register_module("blocks", _blocks);
        register_module("merger", _merger);
        register_module("deepstack_merger_list", _deepstackMergers);
    }

    private Tensor FastPosEmbedInterpolate(Tensor gridThw)
    {
        using var scope = torch.NewDisposeScope();
        var gridThwList = gridThw.data<long>().ToArray();
        var patchEmbedsPerImage = new List<Tensor>();

        for (int i = 0; i < gridThw.shape[0]; i++)
        {
            long t = gridThw[i, 0].item<long>();
            long h = gridThw[i, 1].item<long>();
            long w = gridThw[i, 2].item<long>();

            using var hIdxs = torch.linspace(0, _numGridPerSide - 1, h, device: gridThw.device, dtype: ScalarType.Float32);
            using var wIdxs = torch.linspace(0, _numGridPerSide - 1, w, device: gridThw.device, dtype: ScalarType.Float32);

            using var hFloor = hIdxs.floor().to_type(ScalarType.Int64);
            using var wFloor = wIdxs.floor().to_type(ScalarType.Int64);
            using var hCeil = (hFloor + 1).clamp(max: _numGridPerSide - 1);
            using var wCeil = (wFloor + 1).clamp(max: _numGridPerSide - 1);

            using var dh = hIdxs - hFloor.to_type(ScalarType.Float32);
            using var dw = wIdxs - wFloor.to_type(ScalarType.Float32);

            using var baseH = hFloor * _numGridPerSide;
            using var baseHCeil = hCeil * _numGridPerSide;

            using var idx00 = (baseH.unsqueeze(1) + wFloor.unsqueeze(0)).reshape(-1);
            using var idx01 = (baseH.unsqueeze(1) + wCeil.unsqueeze(0)).reshape(-1);
            using var idx10 = (baseHCeil.unsqueeze(1) + wFloor.unsqueeze(0)).reshape(-1);
            using var idx11 = (baseHCeil.unsqueeze(1) + wCeil.unsqueeze(0)).reshape(-1);

            using var w00 = ((1 - dh).unsqueeze(1) * (1 - dw).unsqueeze(0)).reshape(-1).unsqueeze(1);
            using var w01 = ((1 - dh).unsqueeze(1) * dw.unsqueeze(0)).reshape(-1).unsqueeze(1);
            using var w10 = (dh.unsqueeze(1) * (1 - dw).unsqueeze(0)).reshape(-1).unsqueeze(1);
            using var w11 = (dh.unsqueeze(1) * dw.unsqueeze(0)).reshape(-1).unsqueeze(1);

            var pos00 = _posEmbed.forward(idx00) * w00;
            var pos01 = _posEmbed.forward(idx01) * w01;
            var pos10 = _posEmbed.forward(idx10) * w10;
            var pos11 = _posEmbed.forward(idx11) * w11;
            var posEmbed = pos00 + pos01 + pos10 + pos11;

            if (t > 1)
            {
                posEmbed = posEmbed.repeat(new long[] { t, 1 });
            }

            posEmbed = posEmbed
                .view(t, h / _spatialMergeSize, _spatialMergeSize, w / _spatialMergeSize, _spatialMergeSize, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4);

            patchEmbedsPerImage.Add(scope.MoveToOuter(posEmbed));
        }

        return scope.MoveToOuter(torch.cat(patchEmbedsPerImage, dim: 0));
    }
    private Tensor GetRotPosEmb(Tensor gridThw)
    {
        using var scope = torch.NewDisposeScope();
        var posIdsList = new List<Tensor>();
        int numImages = (int)gridThw.shape[0];

        for (int i = 0; i < numImages; i++)
        {
            long t = gridThw[i, 0].item<long>();
            long h = gridThw[i, 1].item<long>();
            long w = gridThw[i, 2].item<long>();

            var hposIds = torch.arange(h, device: gridThw.device).unsqueeze(1).expand(h, w);
            hposIds = hposIds.reshape(h / _spatialMergeSize, _spatialMergeSize, w / _spatialMergeSize, _spatialMergeSize);
            hposIds = hposIds.permute(0, 2, 1, 3).flatten();

            var wposIds = torch.arange(w, device: gridThw.device).unsqueeze(0).expand(h, w);
            wposIds = wposIds.reshape(h / _spatialMergeSize, _spatialMergeSize, w / _spatialMergeSize, _spatialMergeSize);
            wposIds = wposIds.permute(0, 2, 1, 3).flatten();

            var stacked = torch.stack(new Tensor[] { hposIds, wposIds }, dim: -1);
            posIdsList.Add(stacked.repeat(new long[] { t, 1 }));
        }

        var posIds = torch.cat(posIdsList, dim: 0);
        var maxGridSize = gridThw[.., 1..].max().item<long>();
        var rotaryPosEmbFull = _rotaryPosEmb.forward(maxGridSize);
        var hPosEmb = rotaryPosEmbFull.index_select(0, posIds[.., 0]);
        var wPosEmb = rotaryPosEmbFull.index_select(0, posIds[.., 1]);
        return scope.MoveToOuter(torch.cat(new Tensor[] { hPosEmb, wPosEmb }, dim: -1).flatten(1));
    }

    public (Tensor mergedHiddenStates, List<Tensor> deepstackOutputs) ForwardWithDeepstack(Tensor hiddenStates, Tensor gridThw)
    {
        using var scope = torch.NewDisposeScope();
        hiddenStates = _patchEmbed.forward(hiddenStates);
        hiddenStates = hiddenStates + FastPosEmbedInterpolate(gridThw);
        var rotaryPosEmb = GetRotPosEmb(gridThw);
        var emb = torch.cat(new[] { rotaryPosEmb, rotaryPosEmb }, dim: -1);
        var cos = emb.cos();
        var sin = emb.sin();
        var patchCounts = gridThw[.., 1] * gridThw[.., 2];
        var patchCountsRep = patchCounts.repeat_interleave(gridThw[.., 0]);
        var cuSeqlens = patchCountsRep.to_type(ScalarType.Int64).cumsum(dim: 0);
        cuSeqlens = torch.nn.functional.pad(cuSeqlens, new long[] { 1, 0 }, value: 0);
        var deepstackOutputs = new List<Tensor>();
        for (int layerIdx = 0; layerIdx < _blocks.Count; layerIdx++)
        {
            hiddenStates = _blocks[layerIdx].forward(hiddenStates, cuSeqlens, cos, sin);
            var deepstackIdx = Array.IndexOf(_deepstackVisualIndexes, layerIdx);
            if (deepstackIdx >= 0 && deepstackIdx < _deepstackMergers.Count)
            {
                var deepstackOutput = _deepstackMergers[deepstackIdx].forward(hiddenStates);
                deepstackOutputs.Add(scope.MoveToOuter(deepstackOutput));
            }
        }
        var mergedHiddenStates = scope.MoveToOuter(_merger.forward(hiddenStates));
        return (mergedHiddenStates, deepstackOutputs);
    }
    public Tensor forward(Tensor hiddenStates, Tensor gridThw)
    {
        var (mergedHiddenStates, deepstackOutputs) = ForwardWithDeepstack(hiddenStates, gridThw);
        foreach (var deepstackOutput in deepstackOutputs)
        {
            deepstackOutput.Dispose();
        }
        return mergedHiddenStates;
    }
}

public class Qwen3VLRotaryEmbedding : nn.Module
{
    private readonly Tensor _invFreq;
    private readonly int[] _mRopeSection;

    public Qwen3VLRotaryEmbedding(Qwen3VLTextConfig config, string name = "rotary_emb") : base(name)
    {
        _mRopeSection = config.MRopeSection;
        var freqs = torch.arange((long)0, config.HeadDim, 2, dtype: ScalarType.Float32) / config.HeadDim;
        _invFreq = 1.0 / torch.pow(config.RopeTheta, freqs);
        register_buffer("inv_freq", _invFreq, persistent: false);
    }

    public (Tensor cos, Tensor sin) forward(Tensor x, Tensor positionIds)
    {
        using var scope = torch.NewDisposeScope();

        if (positionIds.ndim == 2)
        {
            positionIds = positionIds.unsqueeze(0).expand(3, positionIds.shape[0], positionIds.shape[1]);
        }

        var invFreqExpanded = _invFreq.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(3, positionIds.shape[1], -1, 1);
        var positionIdsExpanded = positionIds.unsqueeze(2).to(ScalarType.Float32);
        var freqs = torch.matmul(invFreqExpanded, positionIdsExpanded).transpose(2, 3);

        var rope = ApplyInterleavedMrope(freqs, _mRopeSection);
        var emb = torch.cat(new[] { rope, rope }, dim: -1);
        return (scope.MoveToOuter(emb.cos().unsqueeze(1).to_type(x.dtype)), scope.MoveToOuter(emb.sin().unsqueeze(1).to_type(x.dtype)));
    }

    private Tensor ApplyInterleavedMrope(Tensor freqs, int[] mropeSection)
    {
        var freqsT = freqs[0].clone();
        var dimHalf = freqsT.shape[^1];

        for (int dim = 1; dim <= 2; dim++)
        {
            var length = mropeSection[dim] * 3;
            for (int i = dim; i < length; i += 3)
            {
                if (i < dimHalf)
                {
                    freqsT[.., .., i] = freqs[dim, .., .., i];
                }
            }
        }

        return freqsT;
    }
}

public class Qwen3VLAttention : nn.Module
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _numKeyValueHeads;
    private readonly int _numKeyValueGroups;
    private readonly int _layerIdx;
    private readonly Linear _qProj;
    private readonly Linear _kProj;
    private readonly Linear _vProj;
    private readonly Linear _oProj;
    private readonly RMSNorm _qNorm;
    private readonly RMSNorm _kNorm;

    public Qwen3VLAttention(Qwen3VLTextConfig config, int layerIdx, string name = "self_attn") : base(name)
    {
        _hiddenSize = config.HiddenSize;
        _numHeads = config.NumAttentionHeads;
        _headDim = config.HeadDim;
        _numKeyValueHeads = config.NumKeyValueHeads;
        _numKeyValueGroups = _numHeads / _numKeyValueHeads;
        _layerIdx = layerIdx;

        _qProj = Linear(_hiddenSize, _numHeads * _headDim, hasBias: false);
        _kProj = Linear(_hiddenSize, _numKeyValueHeads * _headDim, hasBias: false);
        _vProj = Linear(_hiddenSize, _numKeyValueHeads * _headDim, hasBias: false);
        _oProj = Linear(_numHeads * _headDim, _hiddenSize, hasBias: false);
        _qNorm = new RMSNorm(_headDim, config.RmsNormEps, name: "q_norm");
        _kNorm = new RMSNorm(_headDim, config.RmsNormEps, name: "k_norm");

        register_module("q_proj", _qProj);
        register_module("k_proj", _kProj);
        register_module("v_proj", _vProj);
        register_module("o_proj", _oProj);
        register_module("q_norm", _qNorm);
        register_module("k_norm", _kNorm);
    }

    private Tensor RotateHalf(Tensor x)
    {
        var half = x.shape[^1] / 2;
        var x1 = x[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(null, half)];
        var x2 = x[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(half, null)];
        return torch.cat(new[] { -x2, x1 }, dim: -1);
    }

    public Tensor forward(Tensor hiddenStates, Tensor positionIds, Tensor cos, Tensor sin, Tensor? attentionMask = null, KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();

        long bsz = hiddenStates.shape[0];
        long qLen = hiddenStates.shape[1];

        var queryStates = _qProj.forward(hiddenStates).view(bsz, qLen, _numHeads, _headDim);
        var keyStates = _kProj.forward(hiddenStates).view(bsz, qLen, _numKeyValueHeads, _headDim);
        var valueStates = _vProj.forward(hiddenStates).view(bsz, qLen, _numKeyValueHeads, _headDim).transpose(1, 2);

        queryStates = _qNorm.forward(queryStates).transpose(1, 2);
        keyStates = _kNorm.forward(keyStates).transpose(1, 2);

        queryStates = (queryStates * cos) + (RotateHalf(queryStates) * sin);
        keyStates = (keyStates * cos) + (RotateHalf(keyStates) * sin);

        if (kvCache != null)
        {
            scope.Detach(keyStates);
            scope.Detach(valueStates);
            var kv = kvCache.Update(keyStates, valueStates, _layerIdx);
            keyStates = kv.key;
            valueStates = kv.value;
            scope.Detach(keyStates);
            scope.Detach(valueStates);
        }

        if (_numKeyValueGroups > 1)
        {
            keyStates = keyStates.repeat_interleave(_numKeyValueGroups, dim: 1);
            valueStates = valueStates.repeat_interleave(_numKeyValueGroups, dim: 1);
        }

        bool isCausal = attentionMask is null && qLen > 1;
        var attnOutput = torch.nn.functional.scaled_dot_product_attention(queryStates, keyStates, valueStates, attentionMask, 0.0, isCausal);
        attnOutput = attnOutput.transpose(1, 2).contiguous().view(bsz, qLen, _hiddenSize);
        return scope.MoveToOuter(_oProj.forward(attnOutput));
    }
}

public class Qwen3VLTextMLP : nn.Module
{
    private readonly Linear _gateProj;
    private readonly Linear _upProj;
    private readonly Linear _downProj;

    public Qwen3VLTextMLP(Qwen3VLTextConfig config, string name = "mlp") : base(name)
    {
        _gateProj = Linear(config.HiddenSize, config.IntermediateSize, hasBias: false);
        _upProj = Linear(config.HiddenSize, config.IntermediateSize, hasBias: false);
        _downProj = Linear(config.IntermediateSize, config.HiddenSize, hasBias: false);

        register_module("gate_proj", _gateProj);
        register_module("up_proj", _upProj);
        register_module("down_proj", _downProj);
    }

    public Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        x = torch.nn.functional.silu(_gateProj.forward(x)) * _upProj.forward(x);
        x = _downProj.forward(x);
        return scope.MoveToOuter(x);
    }
}

public class Qwen3VLDecoderLayer : nn.Module
{
    private readonly Qwen3VLAttention _selfAttn;
    private readonly Qwen3VLTextMLP _mlp;
    private readonly RMSNorm _inputLayernorm;
    private readonly RMSNorm _postAttentionLayernorm;

    public Qwen3VLDecoderLayer(Qwen3VLTextConfig config, int layerIdx, string name = "layer") : base(name)
    {
        _selfAttn = new Qwen3VLAttention(config, layerIdx, name: "self_attn");
        _mlp = new Qwen3VLTextMLP(config, name: "mlp");
        _inputLayernorm = new RMSNorm(config.HiddenSize, config.RmsNormEps, name: "input_layernorm");
        _postAttentionLayernorm = new RMSNorm(config.HiddenSize, config.RmsNormEps, name: "post_attention_layernorm");

        register_module("self_attn", _selfAttn);
        register_module("mlp", _mlp);
        register_module("input_layernorm", _inputLayernorm);
        register_module("post_attention_layernorm", _postAttentionLayernorm);
    }

    public Tensor forward(Tensor hiddenStates, Tensor positionIds, Tensor cos, Tensor sin, Tensor? attentionMask = null, KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();
        var residual = hiddenStates;
        hiddenStates = _inputLayernorm.forward(hiddenStates);
        hiddenStates = residual + _selfAttn.forward(hiddenStates, positionIds, cos, sin, attentionMask, kvCache);

        residual = hiddenStates;
        hiddenStates = _postAttentionLayernorm.forward(hiddenStates);
        hiddenStates = residual + _mlp.forward(hiddenStates);
        return scope.MoveToOuter(hiddenStates);
    }
}

public class Qwen3VLTextModel : nn.Module
{
    private readonly Embedding _embedTokens;
    private readonly ModuleList<Qwen3VLDecoderLayer> _layers;
    private readonly RMSNorm _norm;
    private readonly Qwen3VLRotaryEmbedding _rotaryEmb;

    public Qwen3VLTextModel(Qwen3VLConfig config, string name = "language_model") : base(name)
    {
        var textConfig = config.TextConfig;
        _embedTokens = Embedding(textConfig.VocabSize, textConfig.HiddenSize);

        var layers = new List<Qwen3VLDecoderLayer>();
        for (int i = 0; i < textConfig.NumHiddenLayers; i++)
        {
            layers.Add(new Qwen3VLDecoderLayer(textConfig, i, name: $"{i}"));
        }
        _layers = ModuleList(layers.ToArray());
        _norm = new RMSNorm(textConfig.HiddenSize, textConfig.RmsNormEps, name: "norm");
        _rotaryEmb = new Qwen3VLRotaryEmbedding(textConfig, name: "rotary_emb");

        register_module("embed_tokens", _embedTokens);
        register_module("layers", _layers);
        register_module("norm", _norm);
        register_module("rotary_emb", _rotaryEmb);
    }

    public Tensor EmbedTokens(Tensor inputIds) => _embedTokens.forward(inputIds);

    private Tensor DeepstackProcess(Tensor hiddenStates, Tensor visualPosMasks, Tensor visualEmbeds)
    {
        using var scope = torch.NewDisposeScope();
        var output = hiddenStates.clone();
        var mask = visualPosMasks.to(output.device);
        var embeds = visualEmbeds.to(output.device).to_type(output.dtype);
        var indices = mask.nonzero();
        if (indices.shape[0] == embeds.shape[0])
        {
            for (int i = 0; i < indices.shape[0]; i++)
            {
                var batchIdx = indices[i, 0].item<long>();
                var seqIdx = indices[i, 1].item<long>();
                output[batchIdx, seqIdx] = output[batchIdx, seqIdx] + embeds[i];
            }
        }
        return scope.MoveToOuter(output);
    }
    public Tensor forward(
        Tensor inputsEmbeds,
        Tensor positionIds,
        Tensor? attentionMask = null,
        KVCache? kvCache = null,
        Tensor? visualPosMasks = null,
        List<Tensor>? deepstackVisualEmbeds = null)
    {
        using var scope = torch.NewDisposeScope();
        var (cos, sin) = _rotaryEmb.forward(inputsEmbeds, positionIds);
        var hiddenStates = inputsEmbeds;
        for (int layerIdx = 0; layerIdx < _layers.Count; layerIdx++)
        {
            hiddenStates = _layers[layerIdx].forward(hiddenStates, positionIds, cos, sin, attentionMask, kvCache);
            if (visualPosMasks is not null && deepstackVisualEmbeds is not null && layerIdx < deepstackVisualEmbeds.Count)
            {
                hiddenStates = DeepstackProcess(hiddenStates, visualPosMasks, deepstackVisualEmbeds[layerIdx]);
            }
        }
        hiddenStates = _norm.forward(hiddenStates);
        return scope.MoveToOuter(hiddenStates);
    }
}

public class Qwen3VLModel : nn.Module
{
    private readonly Qwen3VLConfig _config;
    private long? _ropeDelta;
    public readonly Qwen3VisionTransformer Visual;
    public readonly Qwen3VLTextModel LanguageModel;

    public Qwen3VLModel(Qwen3VLConfig config, string name = "model") : base(name)
    {
        _config = config;
        Visual = new Qwen3VisionTransformer(config.VisionConfig, name: "visual");
        LanguageModel = new Qwen3VLTextModel(config, name: "language_model");

        register_module("visual", Visual);
        register_module("language_model", LanguageModel);
    }

    private Tensor ComputePositionIds(Tensor inputIds, Tensor inputsEmbeds, KVCache? kvCache, Tensor? imageGridThw)
    {
        if (imageGridThw is not null && (kvCache is null || kvCache.GetSeqLength() == 0))
        {
            var positionIds = QwenVLPositionHelper.BuildPromptPositionIds(
                inputIds,
                _config.ImageTokenId,
                _config.VisionConfig.SpatialMergeSize,
                imageGridThw,
                out var ropeDelta);
            _ropeDelta = ropeDelta;
            return positionIds;
        }

        if (_ropeDelta.HasValue)
        {
            return QwenVLPositionHelper.BuildDecodePositionIds(inputIds, kvCache, _ropeDelta.Value);
        }

        using var scope = torch.NewDisposeScope();
        var bsz = inputsEmbeds.shape[0];
        var seqLen = inputsEmbeds.shape[1];
        var posIds = torch.arange(seqLen, device: inputsEmbeds.device, dtype: ScalarType.Int64).unsqueeze(0).expand(bsz, seqLen);
        return scope.MoveToOuter(posIds.unsqueeze(0).expand(3, bsz, seqLen).clone());
    }

    public Tensor forward(
        Tensor inputIds,
        Tensor? positionIds = null,
        KVCache? kvCache = null,
        Tensor? pixelValues = null,
        Tensor? imageGridThw = null)
    {
        using var scope = torch.NewDisposeScope();
        var inputsEmbeds = LanguageModel.EmbedTokens(inputIds);
        Tensor? visualPosMasks = null;
        List<Tensor>? deepstackVisualEmbeds = null;
        try
        {
            if (pixelValues is not null && imageGridThw is not null)
            {
                var visualOutputs = Visual.ForwardWithDeepstack(pixelValues, imageGridThw);
                var imageEmbeds = visualOutputs.mergedHiddenStates;
                deepstackVisualEmbeds = visualOutputs.deepstackOutputs;
                try
                {
                    var mask = inputIds == _config.ImageTokenId;
                    var indices = mask.nonzero();
                    if (indices.shape[0] != imageEmbeds.shape[0])
                    {
                        throw new InvalidOperationException($"Image features and image tokens do not match, tokens: {indices.shape[0]}, features: {imageEmbeds.shape[0]}");
                    }

                    visualPosMasks = mask;
                    for (int i = 0; i < indices.shape[0]; i++)
                    {
                        var batchIdx = indices[i, 0].item<long>();
                        var seqIdx = indices[i, 1].item<long>();
                        inputsEmbeds[batchIdx, seqIdx] = imageEmbeds[i];
                    }
                }
                finally
                {
                    imageEmbeds.Dispose();
                }
            }
            positionIds ??= ComputePositionIds(inputIds, inputsEmbeds, kvCache, imageGridThw);
            return scope.MoveToOuter(LanguageModel.forward(inputsEmbeds, positionIds, null, kvCache, visualPosMasks, deepstackVisualEmbeds));
        }
        finally
        {
            if (deepstackVisualEmbeds is not null)
            {
                foreach (var deepstackVisualEmbed in deepstackVisualEmbeds)
                {
                    deepstackVisualEmbed.Dispose();
                }
            }
        }
    }
}
public class Qwen3VLForConditionalGeneration : nn.Module, IMultimodalCausalLM
{
    public readonly Qwen3VLModel Model;
    public readonly Linear LmHead;

    public Qwen3VLForConditionalGeneration(Qwen3VLConfig config, string name = "model") : base(name)
    {
        Model = new Qwen3VLModel(config, name: "model");
        LmHead = Linear(config.TextConfig.HiddenSize, config.TextConfig.VocabSize, hasBias: false);

        register_module("model", Model);
        register_module("lm_head", LmHead);
    }

    public Tensor forward(
        Tensor inputIds,
        Tensor? positionIds = null,
        KVCache? kvCache = null,
        Tensor? pixelValues = null,
        Tensor? imageGridThw = null)
    {
        using var scope = torch.NewDisposeScope();
        var hiddenStates = Model.forward(inputIds, positionIds, kvCache, pixelValues, imageGridThw);
        return scope.MoveToOuter(LmHead.forward(hiddenStates));
    }

    Tensor ICausalLM.forward(Tensor inputIds, Tensor? positionIds, KVCache? kvCache)
        => forward(inputIds, positionIds, kvCache, null, null);

    public static Qwen3VLForConditionalGeneration FromPretrained(string modelDirectory)
    {
        var config = LoadConfig(modelDirectory);
        var model = new Qwen3VLForConditionalGeneration(config);
        var stateDict = Zhengyan.QwenSharp.Core.SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        var modelStateDict = model.state_dict();

        var keysToRemove = new List<string>();
        foreach (var (key, tensor) in stateDict)
        {
            if (modelStateDict.TryGetValue(key, out var targetTensor) && !tensor.shape.SequenceEqual(targetTensor.shape))
            {
                keysToRemove.Add(key);
            }
        }

        foreach (var key in keysToRemove)
        {
            stateDict.Remove(key);
        }

        model.load_state_dict(stateDict, strict: false);

        if (config.TextConfig.TieWordEmbeddings)
        {
            var embedWeight = model.get_parameter("model.language_model.embed_tokens.weight");
            var lmHeadWeight = model.LmHead.get_parameter("weight");
            if (embedWeight is not null && lmHeadWeight is not null)
            {
                using var _ = torch.no_grad();
                lmHeadWeight.copy_(embedWeight);
            }
        }

        foreach (var tensor in stateDict.Values)
        {
            tensor.Dispose();
        }

        return model;
    }

    public static Qwen3VLConfig LoadConfig(string modelDirectory)
    {
        var configPath = Path.Combine(modelDirectory, "config.json");
        var configJson = File.ReadAllText(configPath);
        return JsonSerializer.Deserialize<Qwen3VLConfig>(configJson)
               ?? throw new InvalidOperationException("Failed to load Qwen3-VL config.");
    }
}













