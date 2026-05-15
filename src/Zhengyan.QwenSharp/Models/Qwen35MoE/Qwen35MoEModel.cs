using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using Zhengyan.QwenSharp.Models.Qwen35;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen35MoE;

public class Qwen35MoEDecoderLayer : nn.Module
{
    private readonly nn.Module _selfAttn;
    private readonly Qwen35MoESparseMoeBlock _mlp;
    private readonly Qwen35RMSNorm _inputLayernorm;
    private readonly Qwen35RMSNorm _postAttentionLayernorm;
    private readonly string _attentionType;

    public Qwen35MoEDecoderLayer(Qwen35MoEConfig config, int layerIdx, string name = "Qwen35MoEDecoderLayer") : base(name)
    {
        _attentionType = config.LayerTypes != null && layerIdx < config.LayerTypes.Length 
            ? config.LayerTypes[layerIdx] 
            : "full_attention";

        // Convert MoE config to Qwen35Config just for the attention layers
        var attentionConfig = new Qwen35Config
        {
            HiddenSize = config.HiddenSize,
            NumAttentionHeads = config.NumAttentionHeads,
            NumKeyValueHeads = config.NumKeyValueHeads,
            HeadDim = config.HeadDim,
            AttentionBias = config.AttentionBias,
            RmsNormEps = config.RmsNormEps,
            LinearConvKernelDim = config.LinearConvKernelDim,
            LinearKeyHeadDim = config.LinearKeyHeadDim,
            LinearValueHeadDim = config.LinearValueHeadDim,
            LinearNumKeyHeads = config.LinearNumKeyHeads,
            LinearNumValueHeads = config.LinearNumValueHeads
        };

        if (_attentionType == "linear_attention")
        {
            _selfAttn = new Qwen35GatedDeltaNet(attentionConfig, layerIdx, name: "linear_attn");
        }
        else
        {
            _selfAttn = new Qwen35Attention(attentionConfig, layerIdx, name: "self_attn");
        }

        _mlp = new Qwen35MoESparseMoeBlock(config, name: "mlp");
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

public class Qwen35MoEModel : nn.Module
{
    private readonly Embedding _embedTokens;
    private readonly ModuleList<Qwen35MoEDecoderLayer> _layers;
    private readonly Qwen35RMSNorm _norm;
    private readonly Qwen35RotaryEmbedding _rotaryEmb;

    public Qwen35MoEModel(Qwen35MoEConfig config, string name = "model") : base(name)
    {
        _embedTokens = Embedding(config.VocabSize, config.HiddenSize);
        
        _layers = new ModuleList<Qwen35MoEDecoderLayer>();
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            _layers.append(new Qwen35MoEDecoderLayer(config, i, name: $"layers.{i}"));
        }

        _norm = new Qwen35RMSNorm(config.HiddenSize, config.RmsNormEps, name: "norm");
        
        var rotaryConfig = new Qwen35Config { HeadDim = config.HeadDim, PartialRotaryFactor = config.PartialRotaryFactor, RopeTheta = config.RopeTheta };
        _rotaryEmb = new Qwen35RotaryEmbedding(rotaryConfig, name: "rotary_emb");

        register_module("embed_tokens", _embedTokens);
        register_module("layers", _layers);
        register_module("norm", _norm);
        register_module("rotary_emb", _rotaryEmb);
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
        
        var hiddenStates = _embedTokens.forward(inputIds);
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
}

public class Qwen35MoEForCausalLM : nn.Module, ICausalLM
{
    private readonly Qwen35MoEModel _model;
    private readonly Linear _lmHead;

    public Qwen35MoEForCausalLM(Qwen35MoEConfig config, string name = "Qwen35MoEForCausalLM") : base(name)
    {
        _model = new Qwen35MoEModel(config, name: "model");
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

    public static Qwen35MoEForCausalLM FromPretrained(string modelDirectory)
    {
        var configPath = System.IO.Path.Combine(modelDirectory, "config.json");
        var configJson = System.IO.File.ReadAllText(configPath);
        
        using var doc = System.Text.Json.JsonDocument.Parse(configJson);
        string targetJson = configJson;
        if (doc.RootElement.TryGetProperty("text_config", out var textConfigElem))
        {
            targetJson = textConfigElem.GetRawText();
        }

        var config = System.Text.Json.JsonSerializer.Deserialize<Qwen35MoEConfig>(targetJson) 
                     ?? throw new System.InvalidOperationException("Failed to load config.json");

        var model = new Qwen35MoEForCausalLM(config);
        
        var stateDict = Zhengyan.QwenSharp.Core.SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        model.load_state_dict(stateDict, strict: false);
        
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
}
