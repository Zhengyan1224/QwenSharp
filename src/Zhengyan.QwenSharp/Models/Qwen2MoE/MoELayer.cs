using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen2MoE;

public class Qwen2MoETopKRouter : nn.Module
{
    private readonly int _topK;
    private readonly int _numExperts;
    private readonly bool _normTopkProb;
    private readonly long _hiddenDim;
    private readonly Parameter _weight;

    public Qwen2MoETopKRouter(Qwen2MoEConfig config, string name = "Qwen2MoETopKRouter") : base(name)
    {
        _topK = config.NumExpertsPerTok;
        _numExperts = config.NumExperts;
        _normTopkProb = config.NormTopkProb;
        _hiddenDim = config.HiddenSize;
        
        _weight = Parameter(torch.zeros(new long[] { _numExperts, _hiddenDim }));
        
        register_parameter("weight", _weight);
    }

    public (Tensor routerLogits, Tensor routerScores, Tensor routerIndices) forward(Tensor hiddenStates)
    {
        using var scope = torch.NewDisposeScope();
        
        var weight = get_parameter("weight");
        var routerLogits = torch.nn.functional.linear(hiddenStates, weight); // [seq_len, num_experts]
        
        var routerProbs = torch.nn.functional.softmax(routerLogits, dim: -1, dtype: ScalarType.Float32);
        var (routerTopValue, routerIndices) = routerProbs.topk(_topK, dim: -1);
        
        if (_normTopkProb)
        {
            routerTopValue = routerTopValue / routerTopValue.sum(new long[] { -1 }, keepdim: true);
        }
        
        var routerScores = routerTopValue.to(routerLogits.dtype);
        
        return (
            scope.MoveToOuter(routerLogits), 
            scope.MoveToOuter(routerScores), 
            scope.MoveToOuter(routerIndices));
    }
}

public class Qwen2MoEExperts : nn.Module
{
    private readonly int _numExperts;
    private readonly long _hiddenDim;
    private readonly long _intermediateDim;
    private readonly Parameter _gateUpProj;
    private readonly Parameter _downProj;
    private readonly SiLU _actFn;

    public Qwen2MoEExperts(Qwen2MoEConfig config, string name = "Qwen2MoEExperts") : base(name)
    {
        _numExperts = config.NumExperts;
        _hiddenDim = config.HiddenSize;
        _intermediateDim = config.MoeIntermediateSize;
        
        _gateUpProj = Parameter(torch.empty(new long[] { _numExperts, 2 * _intermediateDim, _hiddenDim }));
        _downProj = Parameter(torch.empty(new long[] { _numExperts, _hiddenDim, _intermediateDim }));
        _actFn = SiLU();

        register_parameter("gate_up_proj", _gateUpProj);
        register_parameter("down_proj", _downProj);
    }

    public Tensor forward(Tensor hiddenStates, Tensor topKIndex, Tensor topKWeights)
    {
        using var scope = torch.NewDisposeScope();
        var finalHiddenStates = torch.zeros_like(hiddenStates);

        // expertMask: [batch_size * seq_len, num_experts] -> actually [N, num_experts]
        var expertMask = torch.nn.functional.one_hot(topKIndex, num_classes: _numExperts);
        expertMask = expertMask.permute(2, 1, 0); // [num_experts, top_k, N]
        
        var expertSums = expertMask.sum(new long[] { -1, -2 });
        var expertHit = torch.greater(expertSums, 0).nonzero(); // shape [experts_hit, 1]

        for (int i = 0; i < expertHit.shape[0]; i++)
        {
            var expertIdx = expertHit[i, 0].item<long>();
            if (expertIdx == _numExperts) continue;

            // where returns 2D non-zero indices for the 2D tensor maskForExpert
            var maskForExpert = expertMask[expertIdx]; // shape [top_k, N]
            var nonZero = maskForExpert.nonzero(); // shape [hits, 2]
            
            var topKPos = nonZero[.., 0];
            var tokenIdx = nonZero[.., 1];

            var currentState = hiddenStates[tokenIdx]; // [hits, hiddenDim]
            
            var gateUpProj = get_parameter("gate_up_proj");
            var gateUp = torch.nn.functional.linear(currentState, gateUpProj[expertIdx]);
            
            var chunks = gateUp.chunk(2, dim: -1);
            var gate = chunks[0];
            var up = chunks[1];
            
            var currentHiddenStates = _actFn.forward(gate) * up;
            var downProj = get_parameter("down_proj");
            currentHiddenStates = torch.nn.functional.linear(currentHiddenStates, downProj[expertIdx]);

            var currentWeights = topKWeights[tokenIdx, topKPos].unsqueeze(-1);
            currentHiddenStates = currentHiddenStates * currentWeights;

            finalHiddenStates.index_add_(0, tokenIdx, currentHiddenStates.to(finalHiddenStates.dtype), 1);
        }

        return scope.MoveToOuter(finalHiddenStates);
    }
}

public class Qwen2MoESparseMoeBlock : nn.Module
{
    private readonly Qwen2MoETopKRouter _gate;
    private readonly Qwen2MoEExperts _experts;
    private readonly SwiGLU _sharedExpert;
    private readonly Linear _sharedExpertGate;

    public Qwen2MoESparseMoeBlock(Qwen2MoEConfig config, string name = "mlp") : base(name)
    {
        _gate = new Qwen2MoETopKRouter(config, name: "gate");
        _experts = new Qwen2MoEExperts(config, name: "experts");
        
        _sharedExpert = new SwiGLU(
            config.HiddenSize, 
            config.SharedExpertIntermediateSize, 
            bias: false, 
            name: "shared_expert");
            
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
        var (routerLogits, routingWeights, selectedExperts) = _gate.forward(hiddenStatesReshaped);
        var expertOutput = _experts.forward(hiddenStatesReshaped, selectedExperts, routingWeights);
        
        sharedExpertOutput = torch.nn.functional.sigmoid(_sharedExpertGate.forward(hiddenStatesReshaped)) * sharedExpertOutput;
        
        expertOutput = expertOutput + sharedExpertOutput;
        expertOutput = expertOutput.reshape(batchSize, sequenceLength, hiddenDim);
        
        return scope.MoveToOuter(expertOutput);
    }
}
