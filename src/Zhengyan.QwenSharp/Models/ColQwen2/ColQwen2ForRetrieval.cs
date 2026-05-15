using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Core;
using Zhengyan.QwenSharp.Models.Qwen2;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.ColQwen2;

/// <summary>
/// ColQwen2 Retrieval Model (Late Interaction / ColBERT-style).
/// Computes dense token-level multi-vector representations for queries and documents.
/// Based on the standard Qwen2 architecture.
/// </summary>
public class ColQwen2ForRetrieval : nn.Module
{
    private readonly Qwen2Model _model;
    private readonly Linear _customTextProj;

    public ColQwen2ForRetrieval(Qwen2Config config, string name = "model") : base(name)
    {
        _model = new Qwen2Model(config, name: "model");
        
        // In typical ColBERT architectures, there's a linear projection layer.
        // We project from the LLM hidden size. If ColQwen2 defines a specific dim, 
        // it would be added to the config. By default, it projects back to the hidden size.
        _customTextProj = Linear(config.HiddenSize, config.HiddenSize, hasBias: false);

        register_module("model", _model);
        register_module("custom_text_proj", _customTextProj);
    }

    /// <summary>
    /// Forward pass. Generates L2-normalized dense embeddings for every token in the sequence.
    /// Resulting tensor shape: [batch_size, sequence_length, custom_text_proj_dim]
    /// </summary>
    public Tensor forward(Tensor inputIds, Tensor? positionIds = null, Zhengyan.QwenSharp.Models.Common.KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();
        
        // 1. Get contextualized token embeddings from the Qwen2 LLM backbone
        var hiddenStates = _model.forward(inputIds, positionIds, kvCache, inputsEmbeds: null);
        
        // 2. Compute ColBERT late interaction token embeddings
        var embeddings = _customTextProj.forward(hiddenStates);
        
        // 3. L2 Normalize the vectors along the embedding dimension
        embeddings = nn.functional.normalize(embeddings, p: 2, dim: -1);
        
        return scope.MoveToOuter(embeddings);
    }

    public static ColQwen2ForRetrieval FromPretrained(string modelDirectory)
    {
        var config = ModelConfig.FromDirectory<Qwen2Config>(modelDirectory);
        var model = new ColQwen2ForRetrieval(config);
        
        var stateDict = SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        model.load_state_dict(stateDict, strict: false);
        
        foreach (var tensor in stateDict.Values) tensor.Dispose();
        return model;
    }
}
