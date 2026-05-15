using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using Zhengyan.QwenSharp.Core;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen2;

public class Qwen2DecoderLayer : nn.Module
{
    private readonly Attention _selfAttn;
    private readonly SwiGLU _mlp;
    private readonly RMSNorm _inputLayernorm;
    private readonly RMSNorm _postAttentionLayernorm;

    public Qwen2DecoderLayer(Qwen2Config config, int layerIdx, string name = "Qwen2DecoderLayer") : base(name)
    {
        _selfAttn = new Attention(
            hiddenSize: config.HiddenSize,
            numHeads: config.NumAttentionHeads,
            numKvHeads: config.NumKeyValueHeads,
            qkvBias: true,
            name: "self_attn"
        );

        _mlp = new SwiGLU(
            hiddenSize: config.HiddenSize,
            intermediateSize: config.IntermediateSize,
            bias: false,
            name: "mlp"
        );

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
        hiddenStates = _mlp.forward(hiddenStates);
        hiddenStates = residual + hiddenStates;

        return hiddenStates;
    }
}

public class Qwen2Model : nn.Module
{
    private readonly Embedding _embedTokens;
    private readonly ModuleList<Qwen2DecoderLayer> _layers;
    private readonly RMSNorm _norm;
    private readonly RotaryEmbedding _rotaryEmb;
    private Device[]? _layerDevices;
    private Device? _embedDevice;
    private Device? _normDevice;
    private Device? _rotaryDevice;

    public Qwen2Model(Qwen2Config config, string name = "Qwen2Model") : base(name)
    {
        _embedTokens = Embedding(config.VocabSize, config.HiddenSize);
        
        _layers = new ModuleList<Qwen2DecoderLayer>();
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            _layers.append(new Qwen2DecoderLayer(config, i, name: $"layers.{i}"));
        }

        _norm = new RMSNorm(config.HiddenSize, config.RmsNormEps, name: "norm");
        
        var headDim = (int)(config.HiddenSize / config.NumAttentionHeads);
        _rotaryEmb = new RotaryEmbedding(
            headDim,
            config.MaxPositionEmbeddings,
            config.RopeTheta,
            config.MRopeSection,
            name: "rotary_emb");

        register_module("embed_tokens", _embedTokens);
        register_module("layers", _layers);
        register_module("norm", _norm);
        register_module("rotary_emb", _rotaryEmb);
    }

    /// <summary>
    /// Moves the embedding, decoder layers, normalization, and rotary cache across one or more devices.
    /// When multiple devices are supplied, layers are assigned in contiguous blocks across the list.
    /// </summary>
    public void ApplyDeviceMap(IReadOnlyList<Device> devices)
    {
        if (devices.Count == 0)
        {
            return;
        }

        if (devices.Count == 1)
        {
            var device = devices[0];
            _embedDevice = device;
            _normDevice = device;
            _rotaryDevice = device;
            _layerDevices = Enumerable.Repeat(device, _layers.Count).ToArray();
            ((nn.Module)_embedTokens).to(device);
            ((nn.Module)_norm).to(device);
            ((nn.Module)_rotaryEmb).to(device);
            for (var i = 0; i < _layers.Count; i++)
            {
                ((nn.Module)_layers[i]).to(device);
            }
            return;
        }

        _embedDevice = devices[0];
        _rotaryDevice = devices[0];
        _normDevice = devices[devices.Count - 1];
        _layerDevices = new Device[_layers.Count];

        ((nn.Module)_embedTokens).to(_embedDevice);
        ((nn.Module)_rotaryEmb).to(_rotaryDevice);
        ((nn.Module)_norm).to(_normDevice);

        for (var i = 0; i < _layers.Count; i++)
        {
            var deviceIndex = Math.Min((int)((long)i * devices.Count / _layers.Count), devices.Count - 1);
            var layerDevice = devices[deviceIndex];
            _layerDevices[i] = layerDevice;
            ((nn.Module)_layers[i]).to(layerDevice);
        }
    }

    public Tensor forward(Tensor? inputIds, Tensor? positionIds = null, KVCache? kvCache = null, Tensor? inputsEmbeds = null)
    {
        using var scope = torch.NewDisposeScope();
        
        long seqLen = inputIds is not null ? inputIds.shape[1] : inputsEmbeds.shape[1];
        long batchSize = inputIds is not null ? inputIds.shape[0] : inputsEmbeds.shape[0];
        var device = _embedDevice ?? (inputIds is not null ? inputIds.device : inputsEmbeds.device);

        if (inputIds is not null && inputIds.device != device)
        {
            var movedInputIds = inputIds.to(device);
            inputIds = movedInputIds;
        }

        if (inputsEmbeds is not null && inputsEmbeds.device != device)
        {
            var movedEmbeds = inputsEmbeds.to(device);
            inputsEmbeds = movedEmbeds;
        }

        if (positionIds is null)
        {
            var pastSeqLen = kvCache?.GetSeqLength(0) ?? 0;
            positionIds = torch.arange(pastSeqLen, pastSeqLen + seqLen, device: device, dtype: ScalarType.Int64).unsqueeze(0).expand(batchSize, seqLen);
        }
        else if (positionIds.device != device)
        {
            var movedPositionIds = positionIds.to(device);
            positionIds = movedPositionIds;
        }
        
        var hiddenStates = inputsEmbeds ?? _embedTokens.forward(inputIds);
        var (cos, sin) = _rotaryEmb.forward(hiddenStates, positionIds);
        
        for (int i = 0; i < _layers.Count; i++)
        {
            if (_layerDevices is not null && i < _layerDevices.Length)
            {
                var layerDevice = _layerDevices[i];
                if (hiddenStates.device != layerDevice)
                {
                    var movedHiddenStates = hiddenStates.to(layerDevice);
                    hiddenStates.Dispose();
                    hiddenStates = movedHiddenStates;
                }

                if (positionIds.device != layerDevice)
                {
                    var movedPositionIds = positionIds.to(layerDevice);
                    positionIds.Dispose();
                    positionIds = movedPositionIds;
                }

                if (cos.device != layerDevice)
                {
                    var movedCos = cos.to(layerDevice);
                    cos.Dispose();
                    cos = movedCos;
                }

                if (sin.device != layerDevice)
                {
                    var movedSin = sin.to(layerDevice);
                    sin.Dispose();
                    sin = movedSin;
                }
            }

            hiddenStates = _layers[i].forward(hiddenStates, positionIds, cos, sin, attentionMask: null, kvCache, layerIdx: i);
        }

        if (_normDevice is not null && hiddenStates.device != _normDevice)
        {
            var movedHiddenStates = hiddenStates.to(_normDevice);
            hiddenStates.Dispose();
            hiddenStates = movedHiddenStates;
        }

        hiddenStates = _norm.forward(hiddenStates);
        
        return scope.MoveToOuter(hiddenStates);
    }

    public Embedding GetInputEmbeddings() => _embedTokens;
}

public class Qwen2ForCausalLM : nn.Module, ICausalLM
{
    private readonly Qwen2Model _model;
    private readonly Linear _lmHead;

    public Qwen2ForCausalLM(Qwen2Config config, string name = "Qwen2ForCausalLM") : base(name)
    {
        _model = new Qwen2Model(config, name: "model");
        _lmHead = Linear(config.HiddenSize, config.VocabSize, hasBias: false);

        register_module("model", _model);
        register_module("lm_head", _lmHead);
    }

    public Tensor forward(Tensor inputIds, Tensor? positionIds = null, KVCache? kvCache = null)
    {
        return ForwardInternal(inputIds, positionIds, kvCache, inputsEmbeds: null);
    }

    public Tensor ForwardInternal(Tensor? inputIds, Tensor? positionIds = null, KVCache? kvCache = null, Tensor? inputsEmbeds = null)
    {
        using var scope = torch.NewDisposeScope();
        
        var hiddenStates = _model.forward(inputIds, positionIds, kvCache, inputsEmbeds);
        
        // Generate logits for the sequence
        var logits = _lmHead.forward(hiddenStates);
        
        return scope.MoveToOuter(logits);
    }

    /// <summary>
    /// Loads a Qwen2 model from a HuggingFace pretrained directory.
    /// </summary>
    public static Qwen2ForCausalLM FromPretrained(string modelDirectory)
    {
        var configPath = System.IO.Path.Combine(modelDirectory, "config.json");
        var configJson = System.IO.File.ReadAllText(configPath);
        
        using var doc = System.Text.Json.JsonDocument.Parse(configJson);
        string targetJson = configJson;
        if (doc.RootElement.TryGetProperty("text_config", out var textConfigElem))
        {
            targetJson = textConfigElem.GetRawText();
        }

        var config = System.Text.Json.JsonSerializer.Deserialize<Qwen2Config>(targetJson) 
                     ?? throw new System.InvalidOperationException("Failed to load config.json");

        var model = new Qwen2ForCausalLM(config);
        
        var stateDict = Zhengyan.QwenSharp.Core.SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        model.load_state_dict(stateDict, strict: false); // strict=false to allow missing tie weights or buffers
        
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

        // Free memory used by the loaded dictionary
        foreach (var tensor in stateDict.Values)
        {
            tensor.Dispose();
        }

        return model;
    }
}
