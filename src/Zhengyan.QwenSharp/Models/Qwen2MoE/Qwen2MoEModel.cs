using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Linq;

namespace Zhengyan.QwenSharp.Models.Qwen2MoE;

public class Qwen2MoEDecoderLayer : nn.Module
{
    private readonly Attention _selfAttn;
    private readonly nn.Module _mlp;
    private readonly RMSNorm _inputLayernorm;
    private readonly RMSNorm _postAttentionLayernorm;

    public Qwen2MoEDecoderLayer(Qwen2MoEConfig config, int layerIdx, string name = "Qwen2MoEDecoderLayer") : base(name)
    {
        _selfAttn = new Attention(
            hiddenSize: config.HiddenSize,
            numHeads: config.NumAttentionHeads,
            numKvHeads: config.NumKeyValueHeads,
            qkvBias: true,
            name: "self_attn"
        );

        if (!config.MlpOnlyLayers.Contains(layerIdx) && config.NumExperts > 0 && (layerIdx + 1) % config.DecoderSparseStep == 0)
        {
            _mlp = new Qwen2MoESparseMoeBlock(config, name: "mlp");
        }
        else
        {
            _mlp = new SwiGLU(
                hiddenSize: config.HiddenSize,
                intermediateSize: config.IntermediateSize,
                bias: false,
                name: "mlp"
            );
        }

        _inputLayernorm = new RMSNorm(config.HiddenSize, config.RmsNormEps, name: "input_layernorm");
        _postAttentionLayernorm = new RMSNorm(config.HiddenSize, config.RmsNormEps, name: "post_attention_layernorm");

        register_module("self_attn", _selfAttn);
        register_module("mlp", _mlp);
        register_module("input_layernorm", _inputLayernorm);
        register_module("post_attention_layernorm", _postAttentionLayernorm);
    }

    public Tensor forward(Tensor hiddenStates, Tensor positionIds, Tensor cos, Tensor sin, Tensor? attentionMask = null, KVCache? kvCache = null, int layerIdx = -1)
    {
        var residual = hiddenStates;
        hiddenStates = _inputLayernorm.forward(hiddenStates);
        
        hiddenStates = _selfAttn.forward(hiddenStates, positionIds, cos, sin, attentionMask, kvCache, layerIdx);
        hiddenStates = residual + hiddenStates;

        residual = hiddenStates;
        hiddenStates = _postAttentionLayernorm.forward(hiddenStates);
        
        if (_mlp is Qwen2MoESparseMoeBlock sparseMlp)
        {
            hiddenStates = sparseMlp.forward(hiddenStates);
        }
        else if (_mlp is SwiGLU denseMlp)
        {
            hiddenStates = denseMlp.forward(hiddenStates);
        }
        
        hiddenStates = residual + hiddenStates;

        return hiddenStates;
    }
}

public class Qwen2MoEModel : nn.Module
{
    private readonly Embedding _embedTokens;
    private readonly ModuleList<Qwen2MoEDecoderLayer> _layers;
    private readonly RMSNorm _norm;
    private readonly RotaryEmbedding _rotaryEmb;

    public Qwen2MoEModel(Qwen2MoEConfig config, string name = "model") : base(name)
    {
        _embedTokens = Embedding(config.VocabSize, config.HiddenSize);
        
        _layers = new ModuleList<Qwen2MoEDecoderLayer>();
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            _layers.append(new Qwen2MoEDecoderLayer(config, i, name: $"layers.{i}"));
        }

        _norm = new RMSNorm(config.HiddenSize, config.RmsNormEps, name: "norm");
        
        var headDim = (int)(config.HiddenSize / config.NumAttentionHeads);
        _rotaryEmb = new RotaryEmbedding(headDim, config.MaxPositionEmbeddings, config.RopeTheta, name: "rotary_emb");

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
            hiddenStates = _layers[i].forward(hiddenStates, positionIds, cos, sin, attentionMask: null, kvCache, layerIdx: i);
        }

        hiddenStates = _norm.forward(hiddenStates);
        
        return scope.MoveToOuter(hiddenStates);
    }
}

public class Qwen2MoEForCausalLM : nn.Module, ICausalLM
{
    private readonly Qwen2MoEModel _model;
    private readonly Linear _lmHead;

    public Qwen2MoEForCausalLM(Qwen2MoEConfig config, string name = "Qwen2MoEForCausalLM") : base(name)
    {
        _model = new Qwen2MoEModel(config, name: "model");
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

    public static Qwen2MoEForCausalLM FromPretrained(string modelDirectory)
    {
        var configPath = System.IO.Path.Combine(modelDirectory, "config.json");
        var configJson = System.IO.File.ReadAllText(configPath);
        
        using var doc = System.Text.Json.JsonDocument.Parse(configJson);
        string targetJson = configJson;
        if (doc.RootElement.TryGetProperty("text_config", out var textConfigElem))
        {
            targetJson = textConfigElem.GetRawText();
        }

        var config = System.Text.Json.JsonSerializer.Deserialize<Qwen2MoEConfig>(targetJson) 
                     ?? throw new System.InvalidOperationException("Failed to load config.json");

        var model = new Qwen2MoEForCausalLM(config);
        
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
