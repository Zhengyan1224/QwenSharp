using System;
using System.Collections.Generic;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen35;

public class Qwen35DecoderLayer : nn.Module
{
    private readonly nn.Module _selfAttn;
    private readonly SwiGLU _mlp;
    private readonly Qwen35RMSNorm _inputLayernorm;
    private readonly Qwen35RMSNorm _postAttentionLayernorm;
    private readonly string _attentionType;

    public Qwen35DecoderLayer(Qwen35Config config, int layerIdx, string name = "Qwen35DecoderLayer") : base(name)
    {
        _attentionType = config.LayerTypes != null && layerIdx < config.LayerTypes.Length 
            ? config.LayerTypes[layerIdx] 
            : "full_attention";

        if (_attentionType == "linear_attention")
        {
            _selfAttn = new Qwen35GatedDeltaNet(config, layerIdx, name: "linear_attn");
        }
        else
        {
            _selfAttn = new Qwen35Attention(config, layerIdx, name: "self_attn");
        }

        _mlp = new SwiGLU(config.HiddenSize, config.IntermediateSize, bias: false, name: "mlp");
        _inputLayernorm = new Qwen35RMSNorm(config.HiddenSize, config.RmsNormEps, name: "input_layernorm");
        _postAttentionLayernorm = new Qwen35RMSNorm(config.HiddenSize, config.RmsNormEps, name: "post_attention_layernorm");

        if (_attentionType == "linear_attention")
        {
            register_module("linear_attn", _selfAttn);
        }
        else
        {
            register_module("self_attn", _selfAttn);
        }
        register_module("mlp", _mlp);
        register_module("input_layernorm", _inputLayernorm);
        register_module("post_attention_layernorm", _postAttentionLayernorm);
    }

    public Tensor forward(Tensor hiddenStates, Tensor? positionIds = null, Tensor? cos = null, Tensor? sin = null, Tensor? attentionMask = null, KVCache? kvCache = null, int layerIdx = -1)
    {
        var residual = hiddenStates;
        hiddenStates = _inputLayernorm.forward(hiddenStates);
        
        if (_attentionType == "linear_attention")
        {
            var deltaNet = (Qwen35GatedDeltaNet)_selfAttn;
            hiddenStates = deltaNet.forward(hiddenStates, kvCache, attentionMask);
        }
        else
        {
            var attn = (Qwen35Attention)_selfAttn;
            hiddenStates = attn.forward(hiddenStates, positionIds!, cos!, sin!, attentionMask, kvCache, layerIdx);
        }
        
        hiddenStates = residual + hiddenStates;

        residual = hiddenStates;
        hiddenStates = _postAttentionLayernorm.forward(hiddenStates);
        hiddenStates = _mlp.forward(hiddenStates);
        
        hiddenStates = residual + hiddenStates;

        return hiddenStates;
    }
}

public class Qwen35Model : nn.Module
{
    private readonly Embedding _embedTokens;
    private readonly ModuleList<Qwen35DecoderLayer> _layers;
    private readonly Qwen35RMSNorm _norm;
    private readonly Qwen35RotaryEmbedding _rotaryEmb;

    public Qwen35Model(Qwen35Config config, string name = "model") : base(name)
    {
        _embedTokens = Embedding(config.VocabSize, config.HiddenSize);
        
        _layers = new ModuleList<Qwen35DecoderLayer>();
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            _layers.append(new Qwen35DecoderLayer(config, i, name: $"layers.{i}"));
        }

        _norm = new Qwen35RMSNorm(config.HiddenSize, config.RmsNormEps, name: "norm");
        _rotaryEmb = new Qwen35RotaryEmbedding(config, name: "rotary_emb");

        register_module("embed_tokens", _embedTokens);
        register_module("layers", _layers);
        register_module("norm", _norm);
        register_module("rotary_emb", _rotaryEmb);
    }

    public Tensor GetInputEmbeddings() => _embedTokens.weight;

    public Tensor EmbedTokens(Tensor inputIds) => _embedTokens.forward(inputIds);

    public Tensor ForwardEmbeddings(Tensor inputsEmbeds, Tensor positionIds, KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();

        var hiddenStates = inputsEmbeds;
        var (cos, sin) = _rotaryEmb.forward(hiddenStates, positionIds);

        for (int i = 0; i < _layers.Count; i++)
        {
            hiddenStates = _layers[i].forward(
                hiddenStates,
                positionIds: positionIds,
                cos: cos,
                sin: sin,
                attentionMask: null,
                kvCache: kvCache,
                layerIdx: i
            );
        }

        hiddenStates = _norm.forward(hiddenStates);
        return scope.MoveToOuter(hiddenStates);
    }

    public Tensor forward(Tensor inputIds, Tensor? positionIds = null, KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();
        
        var seqLen = inputIds.shape[1];
        if (positionIds is null)
        {
            var pastSeqLen = kvCache?.GetSeqLength(0) ?? 0;
            positionIds = torch.arange(pastSeqLen, pastSeqLen + seqLen, device: inputIds.device, dtype: ScalarType.Int64).unsqueeze(0).expand(inputIds.shape[0], seqLen);
        }
        
        var hiddenStates = EmbedTokens(inputIds);
        return scope.MoveToOuter(ForwardEmbeddings(hiddenStates, positionIds, kvCache));
    }
}

public class Qwen35ForCausalLM : nn.Module, ICausalLM
{
    private readonly Qwen35Model _model;
    private readonly Linear _lmHead;

    public Qwen35ForCausalLM(Qwen35Config config, string name = "Qwen35ForCausalLM") : base(name)
    {
        _model = new Qwen35Model(config, name: "model");
        _lmHead = Linear(config.HiddenSize, config.VocabSize, hasBias: false);

        register_module("model", _model);
        register_module("lm_head", _lmHead);
    }

    public Tensor forward(Tensor inputIds, Tensor? positionIds = null, KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();
        
        var hiddenStates = _model.forward(inputIds, positionIds, kvCache);
        var logits = _lmHead.forward(hiddenStates);
        
        return scope.MoveToOuter(logits);
    }

    public static Qwen35ForCausalLM FromPretrained(string modelDirectory)
    {
        var config = LoadConfig(modelDirectory);

        // The deserialized config might not capture root elements like "model_type", but it has the architecture dimensions
        var model = new Qwen35ForCausalLM(config);
        
        var stateDict = Zhengyan.QwenSharp.Core.SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        NormalizeStateDictKeys(stateDict);
        model.load_state_dict(stateDict, strict: false);
        
        // Handle tied word embeddings: If the model uses tied embeddings, 
        // the LM head weights are NOT present in the safetensors. 
        // We must manually assign the embedding weight to the LM head to prevent random garbage generation.
        if (config.TieWordEmbeddings)
        {
            var embedWeight = model.get_parameter("model.embed_tokens.weight");
            var lmHeadWeight = model._lmHead.get_parameter("weight");
            if (embedWeight is not null && lmHeadWeight is not null)
            {
                using var _ = torch.no_grad();
                lmHeadWeight.copy_(embedWeight);
            }
        }

        foreach (var tensor in stateDict.Values)
        {
            tensor.Dispose();
        }

        return model;
    }

    public static Qwen35Config LoadConfig(string modelDirectory)
    {
        var configPath = System.IO.Path.Combine(modelDirectory, "config.json");
        var configJson = System.IO.File.ReadAllText(configPath);

        using var doc = System.Text.Json.JsonDocument.Parse(configJson);
        string targetJson = configJson;

        if (doc.RootElement.TryGetProperty("text_config", out var textConfigElem))
        {
            targetJson = textConfigElem.GetRawText();
        }

        var config = System.Text.Json.JsonSerializer.Deserialize<Qwen35Config>(targetJson)
                     ?? throw new System.InvalidOperationException("Failed to load config.json");

        PopulateExtendedConfig(config);
        return config;
    }

    public static void PopulateExtendedConfig(Qwen35Config config)
    {
        if (config.RopeParameters is null)
        {
            return;
        }

        var ropeParameters = config.RopeParameters.Value;
        if (ropeParameters.ValueKind != JsonValueKind.Object)
        {
            return;
        }

        if (ropeParameters.TryGetProperty("rope_theta", out var ropeThetaElem) && ropeThetaElem.ValueKind == JsonValueKind.Number)
        {
            config.RopeTheta = ropeThetaElem.GetDouble();
        }

        if (ropeParameters.TryGetProperty("partial_rotary_factor", out var partialRotaryElem) && partialRotaryElem.ValueKind == JsonValueKind.Number)
        {
            config.PartialRotaryFactor = partialRotaryElem.GetDouble();
        }

        if (ropeParameters.TryGetProperty("mrope_section", out var mropeElem) && mropeElem.ValueKind == JsonValueKind.Array)
        {
            var sections = new List<int>();
            foreach (var section in mropeElem.EnumerateArray())
            {
                if (section.ValueKind == JsonValueKind.Number)
                {
                    sections.Add(section.GetInt32());
                }
            }

            if (sections.Count == 3)
            {
                config.MRopeSection = [.. sections];
            }
        }
    }

    public static void NormalizeStateDictKeys(Dictionary<string, Tensor> stateDict)
    {
        var remappedKeys = new List<(string OldKey, string NewKey)>();

        foreach (var key in stateDict.Keys)
        {
            var normalizedKey = NormalizeStateDictKey(key);
            if (!string.Equals(normalizedKey, key, StringComparison.Ordinal))
            {
                remappedKeys.Add((key, normalizedKey));
            }
        }

        foreach (var (oldKey, newKey) in remappedKeys)
        {
            if (!stateDict.ContainsKey(newKey))
            {
                stateDict[newKey] = stateDict[oldKey];
            }

            stateDict.Remove(oldKey);
        }
    }

    public static string NormalizeStateDictKey(string key)
    {
        if (key.StartsWith("model.language_model.", StringComparison.Ordinal))
        {
            return "model." + key["model.language_model.".Length..];
        }

        return key;
    }
}
