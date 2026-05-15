using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Core;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen35;

/// <summary>
/// Qwen35 Model with a span classification head on top for extractive question-answering 
/// tasks like SQuAD (a linear layer on top of the hidden-states output to compute span start and end logits).
/// </summary>
public class Qwen35ForQuestionAnswering : nn.Module
{
    private readonly Qwen35Model _model;
    private readonly Linear _qaOutputs;

    public Qwen35ForQuestionAnswering(Qwen35Config config, string name = "model") : base(name)
    {
        _model = new Qwen35Model(config, name: "model");
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

    public static Qwen35ForQuestionAnswering FromPretrained(string modelDirectory)
    {
        var config = Qwen35ForCausalLM.LoadConfig(modelDirectory);
        var model = new Qwen35ForQuestionAnswering(config);
        
        var stateDict = SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        Qwen35ForCausalLM.NormalizeStateDictKeys(stateDict);
        model.load_state_dict(stateDict, strict: false);
        
        foreach (var tensor in stateDict.Values) tensor.Dispose();
        return model;
    }
}


