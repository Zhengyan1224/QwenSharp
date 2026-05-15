using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen25VL;

public class Qwen25VLRotaryEmbedding : nn.Module
{
    private readonly Tensor _invFreq;
    private readonly double _attentionScaling;

    public Qwen25VLRotaryEmbedding(Qwen25VLTextConfig config, string name = "rotary_emb") : base(name)
    {
        double baseVal = 1000000.0;
        if (config.RopeScaling != null && config.RopeScaling.TryGetValue("rope_theta", out var thetaObj) && thetaObj is System.Text.Json.JsonElement elem && elem.ValueKind == System.Text.Json.JsonValueKind.Number)
        {
            baseVal = elem.GetDouble();
        }

        long dim = config.HiddenSize / config.NumAttentionHeads;
        _attentionScaling = 1.0;

        var freqs = torch.arange((long)0, dim, 2, dtype: ScalarType.Float32) / dim;
        _invFreq = 1.0 / torch.pow(baseVal, freqs);

        register_buffer("inv_freq", _invFreq, persistent: false);
    }

    public (Tensor cos, Tensor sin) forward(Tensor x, Tensor positionIds)
    {
        using var scope = torch.NewDisposeScope();

        var invFreqExpanded = _invFreq[TensorIndex.None, TensorIndex.None, TensorIndex.Colon, TensorIndex.None].to_type(ScalarType.Float32).expand(3, positionIds.shape[1], -1, 1);
        var positionIdsExpanded = positionIds[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.None, TensorIndex.Colon].to_type(ScalarType.Float32);

        var freqs = torch.matmul(invFreqExpanded, positionIdsExpanded).transpose(2, 3);
        var emb = torch.cat(new[] { freqs, freqs }, dim: -1);

        var cos = emb.cos() * _attentionScaling;
        var sin = emb.sin() * _attentionScaling;

        return (scope.MoveToOuter(cos.to_type(x.dtype)), scope.MoveToOuter(sin.to_type(x.dtype)));
    }
}

public class Qwen25VLAttention : nn.Module
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _numKeyValueHeads;
    private readonly int _numKeyValueGroups;
    private readonly int _layerIdx;
    private readonly Linear _qProj;
    private readonly Linear _kProj;
    private readonly Linear _vProj;
    private readonly Linear _oProj;
    private readonly int[] _mRopeSection;

    public Qwen25VLAttention(Qwen25VLTextConfig config, int layerIdx, string name = "attn") : base(name)
    {
        _hiddenSize = config.HiddenSize;
        _numHeads = config.NumAttentionHeads;
        _headDim = _hiddenSize / _numHeads;
        _numKeyValueHeads = config.NumKeyValueHeads;
        _numKeyValueGroups = _numHeads / _numKeyValueHeads;
        _layerIdx = layerIdx;

        _qProj = Linear(_hiddenSize, _numHeads * _headDim, hasBias: true);
        _kProj = Linear(_hiddenSize, _numKeyValueHeads * _headDim, hasBias: true);
        _vProj = Linear(_hiddenSize, _numKeyValueHeads * _headDim, hasBias: true);
        _oProj = Linear(_numHeads * _headDim, _hiddenSize, hasBias: false);
        _mRopeSection = config.MRopeSection;

        register_module("q_proj", _qProj);
        register_module("k_proj", _kProj);
        register_module("v_proj", _vProj);
        register_module("o_proj", _oProj);
    }

    private Tensor RotateHalf(Tensor x)
    {
        var half = x.shape[^1] / 2;
        var x1 = x[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(null, half)];
        var x2 = x[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(half, null)];
        return torch.cat(new[] { -x2, x1 }, dim: -1);
    }

    public Tensor forward(Tensor hiddenStates, Tensor positionIds, Tensor cos, Tensor sin, Tensor? attentionMask = null, KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();

        long bsz = hiddenStates.shape[0];
        long qLen = hiddenStates.shape[1];

        var queryStates = _qProj.forward(hiddenStates);
        var keyStates = _kProj.forward(hiddenStates);
        var valueStates = _vProj.forward(hiddenStates);

        queryStates = queryStates.view(bsz, qLen, _numHeads, _headDim).transpose(1, 2);
        keyStates = keyStates.view(bsz, qLen, _numKeyValueHeads, _headDim).transpose(1, 2);
        valueStates = valueStates.view(bsz, qLen, _numKeyValueHeads, _headDim).transpose(1, 2);

        var mropeSec = new long[] { _mRopeSection[0] * 2, _mRopeSection[1] * 2, _mRopeSection[2] * 2 };
        var cosSplit = (Tensor[])cos.split(mropeSec, dim: -1);
        var sinSplit = (Tensor[])sin.split(mropeSec, dim: -1);

        var cosParts = new List<Tensor>();
        var sinParts = new List<Tensor>();
        for (int i = 0; i < cosSplit.Length; i++)
        {
            cosParts.Add(cosSplit[i][TensorIndex.Tensor(torch.tensor(i))]);
            sinParts.Add(sinSplit[i][TensorIndex.Tensor(torch.tensor(i))]);
        }

        var cosFinal = torch.cat(cosParts, dim: -1).unsqueeze(1);
        var sinFinal = torch.cat(sinParts, dim: -1).unsqueeze(1);

        queryStates = (queryStates * cosFinal) + (RotateHalf(queryStates) * sinFinal);
        keyStates = (keyStates * cosFinal) + (RotateHalf(keyStates) * sinFinal);

        if (kvCache != null)
        {
            var kv = kvCache.Update(keyStates, valueStates, _layerIdx);
            keyStates = kv.key;
            valueStates = kv.value;
        }

        if (_numKeyValueGroups > 1)
        {
            keyStates = keyStates.repeat_interleave(_numKeyValueGroups, dim: 1);
            valueStates = valueStates.repeat_interleave(_numKeyValueGroups, dim: 1);
        }

        bool isCausal = attentionMask is null && qLen > 1;
        var attnOutput = torch.nn.functional.scaled_dot_product_attention(
            queryStates,
            keyStates,
            valueStates,
            attentionMask,
            0.0,
            isCausal);

        attnOutput = attnOutput.transpose(1, 2).contiguous().view(bsz, qLen, _hiddenSize);
        var output = _oProj.forward(attnOutput);
        return scope.MoveToOuter(output);
    }
}

public class Qwen25VLTextMLP : nn.Module
{
    private readonly Linear _gateProj;
    private readonly Linear _upProj;
    private readonly Linear _downProj;
    private readonly nn.Module<Tensor, Tensor> _act;

    public Qwen25VLTextMLP(Qwen25VLTextConfig config, string name = "mlp") : base(name)
    {
        _gateProj = Linear(config.HiddenSize, config.IntermediateSize, hasBias: false);
        _upProj = Linear(config.HiddenSize, config.IntermediateSize, hasBias: false);
        _downProj = Linear(config.IntermediateSize, config.HiddenSize, hasBias: false);

        _act = config.HiddenAct switch
        {
            "silu" => SiLU(),
            "gelu" => GELU(),
            _ => SiLU()
        };

        register_module("gate_proj", _gateProj);
        register_module("up_proj", _upProj);
        register_module("down_proj", _downProj);
        register_module("act", _act);
    }

    public Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        return scope.MoveToOuter(_downProj.forward(_act.forward(_gateProj.forward(x)) * _upProj.forward(x)));
    }
}

public class Qwen25VLDecoderLayer : nn.Module
{
    private readonly Qwen25VLAttention _selfAttn;
    private readonly Qwen25VLTextMLP _mlp;
    private readonly RMSNorm _inputLayernorm;
    private readonly RMSNorm _postAttentionLayernorm;

    public Qwen25VLDecoderLayer(Qwen25VLTextConfig config, int layerIdx, string name = "layer") : base(name)
    {
        _selfAttn = new Qwen25VLAttention(config, layerIdx, name: "self_attn");
        _mlp = new Qwen25VLTextMLP(config, name: "mlp");
        _inputLayernorm = new RMSNorm(config.HiddenSize, config.RmsNormEps, name: "input_layernorm");
        _postAttentionLayernorm = new RMSNorm(config.HiddenSize, config.RmsNormEps, name: "post_attention_layernorm");

        register_module("self_attn", _selfAttn);
        register_module("mlp", _mlp);
        register_module("input_layernorm", _inputLayernorm);
        register_module("post_attention_layernorm", _postAttentionLayernorm);
    }

    public Tensor forward(Tensor hiddenStates, Tensor positionIds, Tensor cos, Tensor sin, Tensor? attentionMask = null, KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();

        var residual = hiddenStates;
        hiddenStates = _inputLayernorm.forward(hiddenStates);
        hiddenStates = residual + _selfAttn.forward(hiddenStates, positionIds, cos, sin, attentionMask, kvCache);

        residual = hiddenStates;
        hiddenStates = _postAttentionLayernorm.forward(hiddenStates);
        hiddenStates = residual + _mlp.forward(hiddenStates);

        return scope.MoveToOuter(hiddenStates);
    }
}

public class Qwen25VLTextModel : nn.Module
{
    private readonly Embedding _embedTokens;
    private readonly ModuleList<Qwen25VLDecoderLayer> _layers;
    private readonly RMSNorm _norm;
    private readonly Qwen25VLRotaryEmbedding _rotaryEmb;

    public Qwen25VLTextModel(Qwen25VLConfig config, string name = "model") : base(name)
    {
        var textConfig = config.TextConfig;

        _embedTokens = Embedding(textConfig.VocabSize, textConfig.HiddenSize);
        _embedTokens.weight.to_type(ScalarType.Float16);

        var layersList = new List<Qwen25VLDecoderLayer>();
        for (int i = 0; i < textConfig.NumHiddenLayers; i++)
        {
            layersList.Add(new Qwen25VLDecoderLayer(textConfig, i, name: $"{i}"));
        }
        _layers = ModuleList(layersList.ToArray());

        _norm = new RMSNorm(textConfig.HiddenSize, textConfig.RmsNormEps, name: "norm");
        _rotaryEmb = new Qwen25VLRotaryEmbedding(textConfig, name: "rotary_emb");

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

        var (cos, sin) = _rotaryEmb.forward(inputsEmbeds, positionIds);
        var hiddenStates = inputsEmbeds;

        foreach (var layer in _layers)
        {
            hiddenStates = layer.forward(hiddenStates, positionIds, cos, sin, attentionMask, kvCache);
        }

        hiddenStates = _norm.forward(hiddenStates);
        return scope.MoveToOuter(hiddenStates);
    }
}

public class Qwen25VLModel : nn.Module
{
    public readonly Qwen25VisionTransformer Visual;
    public readonly Qwen25VLTextModel LanguageModel;

    public Qwen25VLModel(Qwen25VLConfig config, string name = "model") : base(name)
    {
        Visual = new Qwen25VisionTransformer(config.VisionConfig, name: "visual");
        LanguageModel = new Qwen25VLTextModel(config, name: "language_model");

        register_module("visual", Visual);
        register_module("language_model", LanguageModel);
    }

    public Tensor GetInputEmbeddings() => LanguageModel.GetInputEmbeddings();

    private Tensor Compute3DPositionIds(Tensor inputIds, Tensor inputsEmbeds, Tensor imageGridThw)
    {
        var bsz = inputsEmbeds.shape[0];
        var seqLen = inputsEmbeds.shape[1];
        var posIds = torch.arange(seqLen, device: inputsEmbeds.device).unsqueeze(0).expand(bsz, seqLen);
        return posIds.unsqueeze(0).expand(3, bsz, seqLen);
    }

    public Tensor forward(
        Tensor inputIds = null,
        Tensor? attentionMask = null,
        Tensor positionIds = null,
        Tensor inputsEmbeds = null,
        KVCache? kvCache = null,
        Tensor pixelValues = null,
        Tensor imageGridThw = null)
    {
        using var scope = torch.NewDisposeScope();

        if (inputsEmbeds is null)
        {
            inputsEmbeds = LanguageModel.EmbedTokens(inputIds);
        }

        if (pixelValues is not null)
        {
            var visionOutputs = Visual.forward(pixelValues, imageGridThw);
            var imageEmbeds = visionOutputs;

            if (inputIds is not null)
            {
                var mask = (inputIds == 151655);
                var indices = mask.nonzero();

                if (indices.shape[0] == imageEmbeds.shape[0])
                {
                    for (int i = 0; i < indices.shape[0]; i++)
                    {
                        var bIdx = indices[i, 0].item<long>();
                        var sIdx = indices[i, 1].item<long>();
                        inputsEmbeds[bIdx, sIdx] = imageEmbeds[i];
                    }
                }
            }
        }

        if (positionIds is null)
        {
            positionIds = Compute3DPositionIds(inputIds, inputsEmbeds, imageGridThw);
        }

        var outputs = LanguageModel.forward(inputsEmbeds, positionIds, attentionMask, kvCache);
        return scope.MoveToOuter(outputs);
    }
}

public class Qwen25VLForConditionalGeneration : nn.Module
{
    private readonly Qwen25VLConfig _config;
    public readonly Qwen25VLModel Model;
    public readonly Linear LmHead;

    public Qwen25VLForConditionalGeneration(Qwen25VLConfig config, string name = "model") : base(name)
    {
        _config = config;
        Model = new Qwen25VLModel(config, name: "model");
        LmHead = Linear(config.TextConfig.HiddenSize, config.TextConfig.VocabSize, hasBias: false);

        register_module("model", Model);
        register_module("lm_head", LmHead);
    }

    public Tensor forward(
        Tensor inputIds = null,
        Tensor? attentionMask = null,
        Tensor positionIds = null,
        Tensor inputsEmbeds = null,
        KVCache? kvCache = null,
        Tensor pixelValues = null,
        Tensor imageGridThw = null)
    {
        using var scope = torch.NewDisposeScope();

        var hiddenStates = Model.forward(
            inputIds,
            attentionMask,
            positionIds,
            inputsEmbeds,
            kvCache,
            pixelValues,
            imageGridThw);

        var logits = LmHead.forward(hiddenStates);
        return scope.MoveToOuter(logits);
    }
}
