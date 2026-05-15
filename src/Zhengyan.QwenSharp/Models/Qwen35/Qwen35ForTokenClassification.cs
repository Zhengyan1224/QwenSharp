using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Core;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen35;

/// <summary>
/// Qwen35 Model with a token classification head on top (a linear layer on top of the hidden-states output).
/// e.g. for Named-Entity-Recognition (NER) tasks.
/// </summary>
public class Qwen35ForTokenClassification : nn.Module
{
    private readonly Qwen35Model _model;
    private readonly Dropout _dropout;
    private readonly Linear _score;

    public Qwen35ForTokenClassification(Qwen35Config config, string name = "model") : base(name)
    {
        _model = new Qwen35Model(config, name: "model");
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

    public static Qwen35ForTokenClassification FromPretrained(string modelDirectory)
    {
        var config = Qwen35ForCausalLM.LoadConfig(modelDirectory);
        var model = new Qwen35ForTokenClassification(config);
        
        var stateDict = SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        Qwen35ForCausalLM.NormalizeStateDictKeys(stateDict);
        model.load_state_dict(stateDict, strict: false);
        
        foreach (var tensor in stateDict.Values) tensor.Dispose();
        return model;
    }
}


