using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Core;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen2;

/// <summary>
/// Qwen2 Model with a sequence classification/regression head on top (a linear layer on top of the pooled output).
/// e.g., for GLUE tasks.
/// </summary>
public class Qwen2ForSequenceClassification : nn.Module
{
    private readonly Qwen2Model _model;
    private readonly Linear _score;

    public Qwen2ForSequenceClassification(Qwen2Config config, string name = "model") : base(name)
    {
        _model = new Qwen2Model(config, name: "model");
        _score = Linear(config.HiddenSize, config.NumLabels, hasBias: false);

        register_module("model", _model);
        register_module("score", _score);
    }

    public Tensor forward(Tensor inputIds, Tensor? positionIds = null, Zhengyan.QwenSharp.Models.Common.KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();
        
        var hiddenStates = _model.forward(inputIds, positionIds, kvCache, inputsEmbeds: null);
        var batchSize = hiddenStates.shape[0];
        
        // Select the last token in the sequence dimension (dim 1)
        var lastTokens = hiddenStates.select(1, -1);
        var pooledLogits = _score.forward(lastTokens);
        
        return scope.MoveToOuter(pooledLogits);
    }

    public static Qwen2ForSequenceClassification FromPretrained(string modelDirectory)
    {
        var config = ModelConfig.FromDirectory<Qwen2Config>(modelDirectory);
        var model = new Qwen2ForSequenceClassification(config);
        
        var stateDict = SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        model.load_state_dict(stateDict, strict: false);
        
        foreach (var tensor in stateDict.Values) tensor.Dispose();
        return model;
    }
}
