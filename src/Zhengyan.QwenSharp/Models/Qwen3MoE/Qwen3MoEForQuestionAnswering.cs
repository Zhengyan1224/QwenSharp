using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Core;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen3MoE;

/// <summary>
/// Qwen3MoE Model with a span classification head on top for extractive question-answering 
/// tasks like SQuAD (a linear layer on top of the hidden-states output to compute span start and end logits).
/// </summary>
public class Qwen3MoEForQuestionAnswering : nn.Module
{
    private readonly Qwen3MoEModel _model;
    private readonly Linear _qaOutputs;

    public Qwen3MoEForQuestionAnswering(Qwen3MoEConfig config, string name = "model") : base(name)
    {
        _model = new Qwen3MoEModel(config, name: "model");
        _qaOutputs = Linear(config.HiddenSize, config.NumLabels > 0 ? config.NumLabels : 2, hasBias: true);

        register_module("model", _model);
        register_module("qa_outputs", _qaOutputs);
    }

    /// <summary>
    /// Forward pass returning start and end logits.
    /// Resulting tensor shape: [batch_size, sequence_length, num_labels] (typically num_labels=2).
    /// </summary>
    public Tensor forward(Tensor inputIds, Tensor? positionIds = null, Zhengyan.QwenSharp.Models.Common.KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();
        
        var hiddenStates = _model.forward(inputIds, positionIds, kvCache);
        var logits = _qaOutputs.forward(hiddenStates);
        
        return scope.MoveToOuter(logits);
    }

    public static Qwen3MoEForQuestionAnswering FromPretrained(string modelDirectory)
    {
        var config = ModelConfig.FromDirectory<Qwen3MoEConfig>(modelDirectory);
        var model = new Qwen3MoEForQuestionAnswering(config);
        
        var stateDict = SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        model.load_state_dict(stateDict, strict: false);
        
        foreach (var tensor in stateDict.Values) tensor.Dispose();
        return model;
    }
}


