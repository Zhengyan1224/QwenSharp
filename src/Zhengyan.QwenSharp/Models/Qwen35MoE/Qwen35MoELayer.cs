using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen35MoE;

public class Qwen35MoEExperts : nn.Module
{
    private readonly int _numExperts;
    private readonly long _hiddenDim;
    private readonly long _intermediateDim;

    private readonly Parameter _gateUpProj;
    private readonly Parameter _downProj;

    public Qwen35MoEExperts(Qwen35MoEConfig config, string name = "experts") : base(name)
    {
        _numExperts = config.NumExperts;
        _hiddenDim = config.HiddenSize;
        _intermediateDim = config.MoeIntermediateSize;

        _gateUpProj = Parameter(torch.empty(_numExperts, 2 * _intermediateDim, _hiddenDim));
        _downProj = Parameter(torch.empty(_numExperts, _hiddenDim, _intermediateDim));

        register_parameter("gate_up_proj", _gateUpProj);
        register_parameter("down_proj", _downProj);
    }

    public Tensor forward(Tensor hiddenStates, Tensor topKIndex, Tensor topKWeights)
    {
        using var scope = torch.NewDisposeScope();

        var finalHiddenStates = torch.zeros_like(hiddenStates);

        Tensor expertMask;
        Tensor expertHit;

        using (torch.no_grad())
        {
            expertMask = torch.nn.functional.one_hot(topKIndex, num_classes: _numExperts);
            expertMask = expertMask.permute(2, 1, 0); // [num_experts, top_k, seq_len]
            expertHit = (expertMask.sum(new long[] { -1, -2 }) > 0).nonzero();
        }

        for (int i = 0; i < expertHit.shape[0]; i++)
        {
            var expertIdx = expertHit[i, 0].item<long>();
            if (expertIdx == _numExperts) continue;

            var wh = torch.where(expertMask[expertIdx] > 0);
            var topKPos = wh[0];
            var tokenIdx = wh[1];

            var currentState = hiddenStates.index_select(0, tokenIdx);
            
            var gateUpProj = get_parameter("gate_up_proj");
            var gateUp = torch.nn.functional.linear(currentState, gateUpProj[expertIdx]);
            var chunks = gateUp.chunk(2, dim: -1);
            var gate = chunks[0];
            var up = chunks[1];

            var currentHiddenStates = torch.nn.functional.silu(gate) * up;
            var downProj = get_parameter("down_proj");
            currentHiddenStates = torch.nn.functional.linear(currentHiddenStates, downProj[expertIdx]);
            
            var weightIdx = topKWeights.index(new TensorIndex[] { TensorIndex.Tensor(tokenIdx), TensorIndex.Tensor(topKPos), TensorIndex.None });
            currentHiddenStates = currentHiddenStates * weightIdx;

            finalHiddenStates.index_add_(0, tokenIdx, currentHiddenStates.to(finalHiddenStates.dtype), alpha: 1.0f);
        }

        return scope.MoveToOuter(finalHiddenStates);
    }
}

public class Qwen35MoETopKRouter : nn.Module
{
    private readonly int _topK;
    private readonly int _numExperts;
    private readonly long _hiddenDim;
    private readonly Parameter _weight;

    public Qwen35MoETopKRouter(Qwen35MoEConfig config, string name = "gate") : base(name)
    {
        _topK = config.NumExpertsPerTok;
        _numExperts = config.NumExperts;
        _hiddenDim = config.HiddenSize;

        _weight = Parameter(torch.zeros(_numExperts, _hiddenDim));
        register_parameter("weight", _weight);
    }

    public (Tensor routerLogits, Tensor routerScores, Tensor routerIndices) forward(Tensor hiddenStates)
    {
        using var scope = torch.NewDisposeScope();

        var seqLen = hiddenStates.shape[0];
        hiddenStates = hiddenStates.reshape(-1, _hiddenDim);

        var weight = get_parameter("weight");
        var routerLogits = torch.nn.functional.linear(hiddenStates, weight);
        var routerLogitsSoftmax = torch.nn.functional.softmax(routerLogits, dtype: ScalarType.Float32, dim: -1);
        
        var topKResult = torch.topk(routerLogitsSoftmax, _topK, dim: -1);
        var routerTopValue = topKResult.values;
        var routerIndices = topKResult.indices;

        routerTopValue /= routerTopValue.sum(new long[] { -1 }, keepdim: true);
        routerTopValue = routerTopValue.to(routerLogits.dtype);

        var routerScores = routerTopValue;

        return (scope.MoveToOuter(routerLogits), scope.MoveToOuter(routerScores), scope.MoveToOuter(routerIndices));
    }
}

public class Qwen35MoESparseMoeBlock : nn.Module
{
    private readonly Qwen35MoETopKRouter _gate;
    private readonly Qwen35MoEExperts _experts;
    private readonly SwiGLU _sharedExpert;
    private readonly Linear _sharedExpertGate;

    public Qwen35MoESparseMoeBlock(Qwen35MoEConfig config, string name = "mlp") : base(name)
    {
        _gate = new Qwen35MoETopKRouter(config, name: "gate");
        _experts = new Qwen35MoEExperts(config, name: "experts");
        
        _sharedExpert = new SwiGLU(config.HiddenSize, config.SharedExpertIntermediateSize, bias: false, name: "shared_expert");
        _sharedExpertGate = Linear(config.HiddenSize, 1, hasBias: false);

        register_module("gate", _gate);
        register_module("experts", _experts);
        register_module("shared_expert", _sharedExpert);
        register_module("shared_expert_gate", _sharedExpertGate);
    }

    public Tensor forward(Tensor hiddenStates)
    {
        using var scope = torch.NewDisposeScope();

        var batchSize = hiddenStates.shape[0];
        var sequenceLength = hiddenStates.shape[1];
        var hiddenDim = hiddenStates.shape[2];

        var hiddenStatesReshaped = hiddenStates.view(-1, hiddenDim);

        var sharedExpertOutput = _sharedExpert.forward(hiddenStatesReshaped);
        
        var (_, routingWeights, selectedExperts) = _gate.forward(hiddenStatesReshaped);
        var expertOutput = _experts.forward(hiddenStatesReshaped, selectedExperts, routingWeights);

        var sharedExpertGateOut = torch.nn.functional.sigmoid(_sharedExpertGate.forward(hiddenStatesReshaped).to(ScalarType.Float32)).to(hiddenStates.dtype);
        sharedExpertOutput = sharedExpertGateOut * sharedExpertOutput;

        expertOutput = expertOutput + sharedExpertOutput;
        expertOutput = expertOutput.reshape(batchSize, sequenceLength, hiddenDim);

        return scope.MoveToOuter(expertOutput);
    }
}
