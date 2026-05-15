using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Core;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen2;

/// <summary>
/// Qwen2 Model with a span classification head on top for extractive question-answering 
/// tasks like SQuAD (a linear layer on top of the hidden-states output to compute span start and end logits).
/// </summary>
public class Qwen2ForQuestionAnswering : nn.Module
{
    private readonly Qwen2Model _model;
    private readonly Linear _qaOutputs;

    public Qwen2ForQuestionAnswering(Qwen2Config config, string name = "model") : base(name)
    {
        _model = new Qwen2Model(config, name: "model");
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
        
        var hiddenStates = _model.forward(inputIds, positionIds, kvCache, inputsEmbeds: null);
        var logits = _qaOutputs.forward(hiddenStates);
        
        return scope.MoveToOuter(logits);
    }

    public static Qwen2ForQuestionAnswering FromPretrained(string modelDirectory)
    {
        var config = ModelConfig.FromDirectory<Qwen2Config>(modelDirectory);
        var model = new Qwen2ForQuestionAnswering(config);
        
        var stateDict = SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        model.load_state_dict(stateDict, strict: false);
        
        foreach (var tensor in stateDict.Values) tensor.Dispose();
        return model;
    }
}
