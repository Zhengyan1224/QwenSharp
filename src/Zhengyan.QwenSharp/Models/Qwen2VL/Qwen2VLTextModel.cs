using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen2VL;

public class Qwen2VLAttention : nn.Module
{
    private readonly long _hiddenSize;
    private readonly long _numHeads;
    private readonly long _numKvHeads;
    private readonly long _headDim;
    private readonly long _numKeyValueGroups;

    private readonly Linear _qProj;
    private readonly Linear _kProj;
    private readonly Linear _vProj;
    private readonly Linear _oProj;

    private readonly int[] _mRopeSection;

    public Qwen2VLAttention(Qwen2VLConfig config, int layerIdx = -1, string name = "self_attn") : base(name)
    {
        _hiddenSize = config.HiddenSize;
        _numHeads = config.NumAttentionHeads;
        _numKvHeads = config.NumKeyValueHeads;
        _headDim = _hiddenSize / _numHeads;
        _numKeyValueGroups = _numHeads / _numKvHeads;
        _mRopeSection = config.MRopeSection;

        _qProj = Linear(_hiddenSize, _numHeads * _headDim, hasBias: true);
        _kProj = Linear(_hiddenSize, _numKvHeads * _headDim, hasBias: true);
        _vProj = Linear(_hiddenSize, _numKvHeads * _headDim, hasBias: true);
        _oProj = Linear(_numHeads * _headDim, _hiddenSize, hasBias: false);

        register_module("q_proj", _qProj);
        register_module("k_proj", _kProj);
        register_module("v_proj", _vProj);
        register_module("o_proj", _oProj);
    }

    public Tensor forward(Tensor hiddenStates, Tensor positionIds, Tensor cos, Tensor sin, Tensor? attentionMask = null, KVCache? kvCache = null, int layerIdx = -1)
    {
        using var scope = torch.NewDisposeScope();

        var bsz = hiddenStates.shape[0];
        var qLen = hiddenStates.shape[1];

        var queryStates = _qProj.forward(hiddenStates);
        var keyStates = _kProj.forward(hiddenStates);
        var valueStates = _vProj.forward(hiddenStates);

        queryStates = queryStates.view(bsz, qLen, _numHeads, _headDim).transpose(1, 2);
        keyStates = keyStates.view(bsz, qLen, _numKvHeads, _headDim).transpose(1, 2);
        valueStates = valueStates.view(bsz, qLen, _numKvHeads, _headDim).transpose(1, 2);

        // Apply 3D Rotary Position Embeddings (MRoPE)
        (queryStates, keyStates) = Qwen2VLRotaryEmbedding.ApplyMultimodalRotaryPosEmb(queryStates, keyStates, cos, sin, _mRopeSection);

        // KV Cache
        if (kvCache != null)
        {
            scope.Detach(keyStates);
            scope.Detach(valueStates);
            (keyStates, valueStates) = kvCache.Update(keyStates, valueStates, layerIdx);
            scope.Detach(keyStates);
            scope.Detach(valueStates);
        }

        keyStates = RepeatKV(keyStates, _numKeyValueGroups);
        valueStates = RepeatKV(valueStates, _numKeyValueGroups);

        bool isCausal = attentionMask is null && qLen > 1;
        var attnOutput = torch.nn.functional.scaled_dot_product_attention(
            queryStates, keyStates, valueStates,
            attentionMask,
            0.0,
            isCausal
        );

        attnOutput = attnOutput.transpose(1, 2).contiguous().view(bsz, qLen, _hiddenSize);
        var output = _oProj.forward(attnOutput);

        return scope.MoveToOuter(output);
    }

    private Tensor RepeatKV(Tensor hiddenStates, long nRep)
    {
        if (nRep == 1) return hiddenStates;

        var bsz = hiddenStates.shape[0];
        var numKvHeads = hiddenStates.shape[1];
        var seqLen = hiddenStates.shape[2];
        var headDim = hiddenStates.shape[3];

        var expanded = hiddenStates.unsqueeze(2).expand(bsz, numKvHeads, nRep, seqLen, headDim);
        return expanded.reshape(bsz, numKvHeads * nRep, seqLen, headDim);
    }
}

public class Qwen2VLDecoderLayer : nn.Module
{
    private readonly Qwen2VLAttention _selfAttn;
    private readonly SwiGLU _mlp;
    private readonly RMSNorm _inputLayernorm;
    private readonly RMSNorm _postAttentionLayernorm;

    public Qwen2VLDecoderLayer(Qwen2VLConfig config, int layerIdx, string name = "layer") : base(name)
    {
        _selfAttn = new Qwen2VLAttention(config, layerIdx, name: "self_attn");
        _mlp = new SwiGLU(config.HiddenSize, config.IntermediateSize, name: "mlp");
        _inputLayernorm = new RMSNorm(config.HiddenSize, config.RmsNormEps, name: "input_layernorm");
        _postAttentionLayernorm = new RMSNorm(config.HiddenSize, config.RmsNormEps, name: "post_attention_layernorm");

        register_module("self_attn", _selfAttn);
        register_module("mlp", _mlp);
        register_module("input_layernorm", _inputLayernorm);
        register_module("post_attention_layernorm", _postAttentionLayernorm);
    }

    public Tensor forward(Tensor hiddenStates, Tensor positionIds, Tensor cos, Tensor sin, Tensor? attentionMask = null, KVCache? kvCache = null, int layerIdx = -1)
    {
        using var scope = torch.NewDisposeScope();

        var residual = hiddenStates;
        hiddenStates = _inputLayernorm.forward(hiddenStates);
        hiddenStates = residual + _selfAttn.forward(hiddenStates, positionIds, cos, sin, attentionMask, kvCache, layerIdx);

        residual = hiddenStates;
        hiddenStates = _postAttentionLayernorm.forward(hiddenStates);
        hiddenStates = residual + _mlp.forward(hiddenStates);

        return scope.MoveToOuter(hiddenStates);
    }
}

public class Qwen2VLTextModel : nn.Module
{
    private readonly Qwen2VLConfig _config;
    private readonly Embedding _embedTokens;
    private readonly ModuleList<Qwen2VLDecoderLayer> _layers;
    private readonly RMSNorm _norm;
    private readonly Qwen2VLRotaryEmbedding _rotaryEmb;

    public Qwen2VLTextModel(Qwen2VLConfig config, string name = "language_model") : base(name)
    {
        _config = config;
        _embedTokens = Embedding(config.VocabSize, config.HiddenSize);
        
        var layersList = new List<Qwen2VLDecoderLayer>();
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            layersList.Add(new Qwen2VLDecoderLayer(config, i, name: $"{i}"));
        }
        _layers = ModuleList(layersList.ToArray());

        _norm = new RMSNorm(config.HiddenSize, config.RmsNormEps, name: "norm");
        _rotaryEmb = new Qwen2VLRotaryEmbedding(config, name: "rotary_emb");

        register_module("embed_tokens", _embedTokens);
        register_module("layers", _layers);
        register_module("norm", _norm);
        register_module("rotary_emb", _rotaryEmb);
    }

    public Tensor GetInputEmbeddings() => _embedTokens.weight;
    
    public Tensor EmbedTokens(Tensor inputIds) => _embedTokens.forward(inputIds);

    public Tensor forward(Tensor inputsEmbeds, Tensor positionIds, Tensor? attentionMask = null, KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();
        
        var hiddenStates = inputsEmbeds;
        
        var (cos, sin) = _rotaryEmb.forward(hiddenStates, positionIds);

        // For positionIds to pass into the decoder layers, we only need the 1D part to derive mask causality,
        // but since causal is derived from QLen in standard Attention without complex padding, we may just pass the full positionIds.
        var textPositionIds = positionIds.ndim == 3 ? positionIds[0] : positionIds;

        for (int i = 0; i < _layers.Count; i++)
        {
            hiddenStates = _layers[i].forward(hiddenStates, textPositionIds, cos, sin, attentionMask, kvCache, i);
        }

        hiddenStates = _norm.forward(hiddenStates);
        
        return scope.MoveToOuter(hiddenStates);
    }
}
