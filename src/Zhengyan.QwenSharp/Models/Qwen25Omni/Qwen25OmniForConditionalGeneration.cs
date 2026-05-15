using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Core;
using Zhengyan.QwenSharp.Models.Common;
using Zhengyan.QwenSharp.Models;
using Zhengyan.QwenSharp.Models.Qwen2;
using Zhengyan.QwenSharp.Models.Qwen2Audio;
using Zhengyan.QwenSharp.Models.Qwen25VL;
using Zhengyan.QwenSharp.Models.Vision;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen25Omni;

/// <summary>
/// Adapter that converts a Qwen25OmniAudioEncoderConfig into the Qwen2AudioEncoderConfig
/// format used by the existing Qwen2AudioEncoder implementation.
/// </summary>
internal static class Qwen25OmniAudioConfigAdapter
{
    public static Qwen2AudioEncoderConfig ToQwen2AudioEncoderConfig(Qwen25OmniAudioEncoderConfig src)
    {
        return new Qwen2AudioEncoderConfig
        {
            NumMelBins = src.NumMelBins,
            DModel = src.DModel,
            EncoderLayers = src.EncoderLayers,
            EncoderAttentionHeads = src.EncoderAttentionHeads,
            EncoderFfnDim = src.EncoderFfnDim,
            ActivationFunction = src.ActivationFunction,
            MaxSourcePositions = src.MaxSourcePositions,
            NWindow = src.NWindow,
            OutputDim = src.OutputDim,
            UseOmniAudioChunking = true,
            UseSinusoidalPositionEmbedding = true,
            ProjectOutput = true,
        };
    }

    public static Qwen2Config ToQwen2TextConfig(Qwen25OmniTextConfig src)
    {
        return new Qwen2Config
        {
            VocabSize = src.VocabSize,
            HiddenSize = (long)src.HiddenSize,
            IntermediateSize = (long)src.IntermediateSize,
            NumHiddenLayers = src.NumHiddenLayers,
            NumAttentionHeads = src.NumAttentionHeads,
            NumKeyValueHeads = src.NumKeyValueHeads ?? src.NumAttentionHeads,
            RmsNormEps = src.RmsNormEps,
            RopeTheta = src.RopeTheta,
            MaxPositionEmbeddings = src.MaxPositionEmbeddings,
            AttentionDropout = src.AttentionDropout,
            UseSlidingWindow = src.UseSlidingWindow,
            SlidingWindow = src.SlidingWindow,
            MaxWindowLayers = src.MaxWindowLayers,
            TieWordEmbeddings = src.TieWordEmbeddings,
            MRopeSection = src.MRopeSection,
        };
    }

    public static Qwen25VLVisionConfig ToQwen25VLVisionConfig(Qwen25OmniVisionEncoderConfig src)
    {
        return new Qwen25VLVisionConfig
        {
            Depth = src.Depth,
            HiddenSize = src.HiddenSize,
            HiddenAct = src.HiddenAct,
            IntermediateSize = src.IntermediateSize,
            NumHeads = src.NumHeads,
            InChannels = src.InChannels,
            PatchSize = src.PatchSize,
            SpatialMergeSize = src.SpatialMergeSize,
            TemporalPatchSize = src.TemporalPatchSize,
            WindowSize = src.WindowSize,
            OutHiddenSize = src.OutHiddenSize,
            FullattBlockIndexes = src.FullattBlockIndexes,
        };
    }
}

/// <summary>
/// Multimodal projector from audio encoder output dimension to text hidden size.
/// </summary>
public class Qwen25OmniAudioProjector : nn.Module
{
    private readonly Linear _linear;

    public Qwen25OmniAudioProjector(int audioDim, int textHiddenSize, string name = "audio_projection") : base(name)
    {
        _linear = Linear(audioDim, textHiddenSize, hasBias: true);
        register_module("linear", _linear);
    }

    public Tensor forward(Tensor audioFeatures)
    {
        using var scope = torch.NewDisposeScope();
        return scope.MoveToOuter(_linear.forward(audioFeatures));
    }
}

/// <summary>
/// The unified Qwen2.5-Omni Thinker model for multimodal conditional generation.
/// Integrates:
/// - Audio Encoder (Qwen2Audio-style Whisper encoder)
/// - Audio-to-LLM projectior
/// - Text Decoder (Qwen2-style causal LLM)
///
/// Vision support is mapped from Qwen2.5-VL but is left as future work pending
/// the full Zhengyan.QwenSharp Qwen2.5-VL vision encoder integration.
/// </summary>
public class Qwen25OmniThinkerForConditionalGeneration : nn.Module, ICausalLM
{
    private readonly Qwen25OmniThinkerConfig _config;
    private readonly Qwen2AudioEncoder _audioTower;
    private readonly Qwen25OmniAudioProjector? _audioProjector;
    private readonly Qwen25VisionTransformer _visionTower;
    private readonly Linear? _visionProjector;
    private readonly Qwen2Model _languageModel;
    private readonly Linear _lmHead;

    public Qwen25OmniThinkerForConditionalGeneration(
        Qwen25OmniThinkerConfig config,
        string name = "thinker") : base(name)
    {
        _config = config;

        // Build audio encoder using existing Qwen2Audio infrastructure
        var audioEncoderConfig = Qwen25OmniAudioConfigAdapter.ToQwen2AudioEncoderConfig(config.AudioConfig);
        _audioTower = new Qwen2AudioEncoder(audioEncoderConfig, name: "audio_tower");

        // Qwen2.5-Omni's audio tower already owns audio_tower.proj. Keep this
        // fallback only for configs whose audio output dimension differs.
        var audioOutputDim = config.AudioConfig.OutputDim > 0 ? config.AudioConfig.OutputDim : config.AudioConfig.DModel;
        if (audioOutputDim != config.TextConfig.HiddenSize)
        {
            _audioProjector = new Qwen25OmniAudioProjector(
                audioDim: audioOutputDim,
                textHiddenSize: config.TextConfig.HiddenSize,
                name: "audio_projection");
        }

        _visionTower = new Qwen25VisionTransformer(
            Qwen25OmniAudioConfigAdapter.ToQwen25VLVisionConfig(config.VisionConfig),
            name: "visual");
        if (config.VisionConfig.OutHiddenSize != config.TextConfig.HiddenSize)
        {
            _visionProjector = Linear(config.VisionConfig.OutHiddenSize, config.TextConfig.HiddenSize, hasBias: true);
        }

        // Build text decoder using existing Qwen2 infrastructure
        var textConfig = Qwen25OmniAudioConfigAdapter.ToQwen2TextConfig(config.TextConfig);
        _languageModel = new Qwen2Model(textConfig, name: "model");

        // LM head
        _lmHead = Linear(config.TextConfig.HiddenSize, config.TextConfig.VocabSize, hasBias: false);

        register_module("audio_tower", _audioTower);
        if (_audioProjector is not null)
        {
            register_module("audio_projection", _audioProjector);
        }
        register_module("visual", _visionTower);
        if (_visionProjector is not null)
        {
            register_module("vision_projection", _visionProjector);
        }
        register_module("model", _languageModel);
        register_module("lm_head", _lmHead);
    }

    /// <summary>
    /// Moves the Omni thinker across one or more devices.
    /// When multiple devices are supplied, the audio/vision front-end stays on the first device,
    /// the decoder layers are sharded across the list, and the LM head ends up on the last device.
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
            ((nn.Module)_audioTower).to(device);
            if (_audioProjector is not null)
            {
                ((nn.Module)_audioProjector).to(device);
            }
            ((nn.Module)_visionTower).to(device);
            if (_visionProjector is not null)
            {
                ((nn.Module)_visionProjector).to(device);
            }
            _languageModel.ApplyDeviceMap(devices);
            ((nn.Module)_lmHead).to(device);
            return;
        }

        var firstDevice = devices[0];
        var lastDevice = devices[devices.Count - 1];

        ((nn.Module)_audioTower).to(firstDevice);
        if (_audioProjector is not null)
        {
            ((nn.Module)_audioProjector).to(firstDevice);
        }
        ((nn.Module)_visionTower).to(firstDevice);
        if (_visionProjector is not null)
        {
            ((nn.Module)_visionProjector).to(firstDevice);
        }
        _languageModel.ApplyDeviceMap(devices);
        ((nn.Module)_lmHead).to(lastDevice);
    }

    /// <summary>
    /// Text-only forward pass implementing ICausalLM interface.
    /// </summary>
    public Tensor forward(Tensor inputIds, Tensor? positionIds = null, KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();
        var hiddenStates = _languageModel.forward(inputIds, positionIds, kvCache);
        var logits = _lmHead.forward(hiddenStates);
        return scope.MoveToOuter(logits);
    }

    /// <summary>
    /// Multimodal forward: merges audio features into text embeddings before decoding.
    /// </summary>
    public Tensor forward(
        Tensor inputIds,
        Tensor inputFeatures,
        Tensor? positionIds = null,
        KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();

        // 1. Get text token embeddings
        var inputsEmbeds = _languageModel.GetInputEmbeddings().forward(inputIds);

        // 2. Extract audio features from mel-spectrogram
        //    audioOutputs: [batch, audio_seq_len, d_model]
        var audioOutputs = _audioTower.forward(inputFeatures);

        // 3. Project audio to text hidden dimension if the audio tower did not already do it.
        var audioFeaturesProjected = _audioProjector is not null
            ? _audioProjector.forward(audioOutputs)
            : audioOutputs;

        // 4. Replace audio placeholder tokens with projected audio embeddings
        var audioTokenMask = inputIds == _config.AudioTokenIndex;
        var audioTokenCount = audioTokenMask.sum().item<long>();
        var expectedCount = audioFeaturesProjected.ndim == 3
            ? audioFeaturesProjected.shape[0] * audioFeaturesProjected.shape[1]
            : audioFeaturesProjected.shape[0];

        if (audioTokenCount > 0 && audioTokenCount == expectedCount)
        {
            inputsEmbeds[audioTokenMask] = audioFeaturesProjected
                .view(-1, audioFeaturesProjected.shape[^1])
                .to_type(inputsEmbeds.dtype);
        }
        else if (audioTokenCount > 0)
        {
            Console.WriteLine(
                $"Warning: Audio token count in input ({audioTokenCount}) " +
                $"doesn't match expected ({expectedCount}). Skipping audio injection.");
        }

        // 5. Forward through decoder with merged embeddings
        var hiddenStates = _languageModel.forward(
            inputIds: null,
            positionIds: positionIds,
            kvCache: kvCache,
            inputsEmbeds: inputsEmbeds);

        var logits = _lmHead.forward(hiddenStates);
        return scope.MoveToOuter(logits);
    }

    public Tensor forward(
        Tensor inputIds,
        Tensor? positionIds = null,
        KVCache? kvCache = null,
        Tensor? visionPixelValues = null,
        Tensor? visionGridThw = null)
    {
        if (visionPixelValues is null || visionGridThw is null)
        {
            return forward(inputIds, positionIds, kvCache);
        }

        using var scope = torch.NewDisposeScope();
        var inputsEmbeds = _languageModel.GetInputEmbeddings().forward(inputIds);
        var visionOutputs = _visionTower.forward(visionPixelValues, visionGridThw);
        var visionProjected = _visionProjector is not null
            ? _visionProjector.forward(visionOutputs)
            : visionOutputs;
        var visionTokenMask = (inputIds == _config.ImageTokenIndex) | (inputIds == _config.VideoTokenIndex);
        var visionTokenCount = visionTokenMask.sum().item<long>();

        if (visionTokenCount == visionProjected.shape[0])
        {
            inputsEmbeds[visionTokenMask] = visionProjected.view(-1, visionProjected.shape[^1]).to_type(inputsEmbeds.dtype);
        }
        else if (visionTokenCount > 0)
        {
            Console.WriteLine(
                $"Warning: Vision token count in input ({visionTokenCount}) " +
                $"doesn't match expected ({visionProjected.shape[0]}). Skipping vision injection.");
        }

        var hiddenStates = _languageModel.forward(
            inputIds: null,
            positionIds: positionIds,
            kvCache: kvCache,
            inputsEmbeds: inputsEmbeds);

        var logits = _lmHead.forward(hiddenStates);
        return scope.MoveToOuter(logits);
    }

    public Embedding GetInputEmbeddings() => _languageModel.GetInputEmbeddings();
}

/// <summary>
/// Top-level Qwen2.5-Omni model container wrapping the Thinker sub-model.
/// Supports loading from a pretrained directory.
/// </summary>
public class Qwen25OmniForConditionalGeneration : nn.Module, IMultimodalCausalLM
{
    private readonly Qwen25OmniConfig _config;
    public readonly Qwen25OmniThinkerForConditionalGeneration Thinker;

    public Qwen25OmniForConditionalGeneration(Qwen25OmniConfig config, string name = "model") : base(name)
    {
        _config = config;
        Thinker = new Qwen25OmniThinkerForConditionalGeneration(config.ThinkerConfig, name: "thinker");
        register_module("thinker", Thinker);
    }

    public void ApplyDeviceMap(IReadOnlyList<Device> devices)
        => Thinker.ApplyDeviceMap(devices);

    /// <summary>
    /// Text-only forward (implements ICausalLM).
    /// </summary>
    public Tensor forward(Tensor inputIds, Tensor? positionIds = null, KVCache? kvCache = null)
        => Thinker.forward(inputIds, positionIds, kvCache);

    /// <summary>
    /// Multimodal forward with audio features.
    /// </summary>
    public Tensor forward(
        Tensor inputIds,
        Tensor inputFeatures,
        Tensor? positionIds = null,
        KVCache? kvCache = null)
        => Thinker.forward(inputIds, inputFeatures, positionIds, kvCache);

    public Tensor forward(
        Tensor inputIds,
        Tensor? positionIds = null,
        KVCache? kvCache = null,
        Tensor? visionPixelValues = null,
        Tensor? visionGridThw = null)
        => Thinker.forward(inputIds, positionIds, kvCache, visionPixelValues, visionGridThw);

    /// <summary>
    /// Loads model from a pretrained directory containing config.json and safetensors weights.
    /// </summary>
    public static Qwen25OmniForConditionalGeneration FromPretrained(string modelDirectory)
    {
        var config = ModelConfig.FromDirectory<Qwen25OmniConfig>(modelDirectory);
        var model = new Qwen25OmniForConditionalGeneration(config);

        var stateDict = Core.SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        var normalizedStateDict = NormalizeStateDictKeys(stateDict);
        model.load_state_dict(normalizedStateDict, strict: false);

        var originalTensors = new HashSet<Tensor>(stateDict.Values);
        foreach (var tensor in normalizedStateDict.Values)
        {
            if (!originalTensors.Contains(tensor))
            {
                tensor.Dispose();
            }
        }

        foreach (var tensor in stateDict.Values)
            tensor.Dispose();

        return model;
    }

    private static Dictionary<string, Tensor> NormalizeStateDictKeys(Dictionary<string, Tensor> stateDict)
    {
        var normalized = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        var remappedCount = 0;

        foreach (var (key, tensor) in stateDict)
        {
            var normalizedKey = NormalizeStateDictKey(key);
            if (!string.Equals(key, normalizedKey, StringComparison.Ordinal))
            {
                remappedCount++;
            }

            if (!normalized.ContainsKey(normalizedKey) ||
                string.Equals(key, normalizedKey, StringComparison.Ordinal))
            {
                normalized[normalizedKey] = tensor;
            }
        }

        if (remappedCount > 0)
        {
            Console.WriteLine($"Qwen2.5-Omni weight key normalization remapped {remappedCount} tensors.");
        }

        var fusedVisionQkvCount = AddFusedVisionQkvWeights(normalized);
        if (fusedVisionQkvCount > 0)
        {
            Console.WriteLine($"Qwen2.5-Omni fused {fusedVisionQkvCount} visual q/k/v tensor group(s) into qkv tensors.");
        }

        WarnIfMissingCriticalWeights(normalized);
        return normalized;
    }

    private static int AddFusedVisionQkvWeights(Dictionary<string, Tensor> stateDict)
    {
        var fusedCount = 0;
        var qWeightSuffix = ".attn.q.weight";
        var qBiasSuffix = ".attn.q.bias";

        foreach (var qKey in stateDict.Keys.Where(key => key.EndsWith(qWeightSuffix, StringComparison.Ordinal)).ToArray())
        {
            var prefix = qKey[..^qWeightSuffix.Length];
            var targetKey = prefix + ".attn.qkv.weight";
            if (stateDict.ContainsKey(targetKey))
            {
                continue;
            }

            var kKey = prefix + ".attn.k.weight";
            var vKey = prefix + ".attn.v.weight";
            if (!stateDict.TryGetValue(qKey, out var qWeight) ||
                !stateDict.TryGetValue(kKey, out var kWeight) ||
                !stateDict.TryGetValue(vKey, out var vWeight))
            {
                continue;
            }

            stateDict[targetKey] = torch.cat(new[] { qWeight, kWeight, vWeight }, dim: 0);
            fusedCount++;
        }

        foreach (var qKey in stateDict.Keys.Where(key => key.EndsWith(qBiasSuffix, StringComparison.Ordinal)).ToArray())
        {
            var prefix = qKey[..^qBiasSuffix.Length];
            var targetKey = prefix + ".attn.qkv.bias";
            if (stateDict.ContainsKey(targetKey))
            {
                continue;
            }

            var kKey = prefix + ".attn.k.bias";
            var vKey = prefix + ".attn.v.bias";
            if (!stateDict.TryGetValue(qKey, out var qBias) ||
                !stateDict.TryGetValue(kKey, out var kBias) ||
                !stateDict.TryGetValue(vKey, out var vBias))
            {
                continue;
            }

            stateDict[targetKey] = torch.cat(new[] { qBias, kBias, vBias }, dim: 0);
            fusedCount++;
        }

        return fusedCount;
    }

    private static string NormalizeStateDictKey(string key)
    {
        if (key.StartsWith("model.thinker.", StringComparison.Ordinal))
        {
            key = key["model.".Length..];
        }

        if (key.StartsWith("thinker.language_model.", StringComparison.Ordinal))
        {
            return "thinker.model." + key["thinker.language_model.".Length..];
        }

        if (key.StartsWith("thinker.vision_tower.", StringComparison.Ordinal))
        {
            return "thinker.visual." + key["thinker.vision_tower.".Length..];
        }

        return key;
    }

    private static void WarnIfMissingCriticalWeights(IReadOnlyDictionary<string, Tensor> stateDict)
    {
        var criticalKeys = new[]
        {
            "thinker.model.embed_tokens.weight",
            "thinker.lm_head.weight",
            "thinker.audio_tower.conv1.weight",
        };

        foreach (var key in criticalKeys)
        {
            if (!stateDict.ContainsKey(key))
            {
                Console.WriteLine($"Warning: expected Qwen2.5-Omni checkpoint tensor '{key}' was not found.");
            }
        }
    }
}
