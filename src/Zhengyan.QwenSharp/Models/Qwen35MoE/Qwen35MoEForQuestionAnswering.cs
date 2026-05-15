using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Core;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen35MoE;

/// <summary>
/// Qwen35MoE Model with a span classification head on top for extractive question-answering 
/// tasks like SQuAD (a linear layer on top of the hidden-states output to compute span start and end logits).
/// </summary>
public class Qwen35MoEForQuestionAnswering : nn.Module
{
    private readonly Qwen35MoEModel _model;
    private readonly Linear _qaOutputs;

    public Qwen35MoEForQuestionAnswering(Qwen35MoEConfig config, string name = "model") : base(name)
    {
        _model = new Qwen35MoEModel(config, name: "model");
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

    public static Qwen35MoEForQuestionAnswering FromPretrained(string modelDirectory)
    {
        var config = ModelConfig.FromDirectory<Qwen35MoEConfig>(modelDirectory);
        var model = new Qwen35MoEForQuestionAnswering(config);
        
        var stateDict = SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        model.load_state_dict(stateDict, strict: false);
        
        foreach (var tensor in stateDict.Values) tensor.Dispose();
        return model;
    }
}


