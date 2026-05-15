using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Core;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen35MoE;

/// <summary>
/// Qwen35MoE Model with a token classification head on top (a linear layer on top of the hidden-states output).
/// e.g. for Named-Entity-Recognition (NER) tasks.
/// </summary>
public class Qwen35MoEForTokenClassification : nn.Module
{
    private readonly Qwen35MoEModel _model;
    private readonly Dropout _dropout;
    private readonly Linear _score;

    public Qwen35MoEForTokenClassification(Qwen35MoEConfig config, string name = "model") : base(name)
    {
        _model = new Qwen35MoEModel(config, name: "model");
        _dropout = Dropout(config.ClassifierDropout ?? config.AttentionDropout);
        // Default in HF token classification is hasBias=True
        _score = Linear(config.HiddenSize, config.NumLabels, hasBias: true);

        // Register with same nested name structure as HuggingFace Transformers
        register_module("model", _model);
        register_module("score", _score);
    }

    public Tensor forward(Tensor inputIds, Tensor? positionIds = null, Zhengyan.QwenSharp.Models.Common.KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();
        
        var hiddenStates = _model.forward(inputIds, positionIds, kvCache);
        
        // Apply dropout over [batch_size, seq_len, hidden_size]
        hiddenStates = _dropout.forward(hiddenStates);
        
        // Logits shape: [batch_size, seq_len, num_labels]
        var logits = _score.forward(hiddenStates);
        
        return scope.MoveToOuter(logits);
    }

    public static Qwen35MoEForTokenClassification FromPretrained(string modelDirectory)
    {
        var config = ModelConfig.FromDirectory<Qwen35MoEConfig>(modelDirectory);
        var model = new Qwen35MoEForTokenClassification(config);
        
        var stateDict = SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        model.load_state_dict(stateDict, strict: false);
        
        foreach (var tensor in stateDict.Values) tensor.Dispose();
        return model;
    }
}


