using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Core;
using Zhengyan.QwenSharp.Models.Common;
using Zhengyan.QwenSharp.Models.Qwen2;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen2Audio;

public class Qwen2AudioMultiModalProjector : nn.Module
{
    private readonly Linear _linear;

    public Qwen2AudioMultiModalProjector(Qwen2AudioConfig config, string name = "multi_modal_projector") : base(name)
    {
        _linear = Linear(config.AudioConfig.DModel, config.TextConfig.HiddenSize, hasBias: true);
        register_module("linear", _linear);
    }

    public Tensor forward(Tensor audioFeatures)
    {
        using var scope = torch.NewDisposeScope();
        return scope.MoveToOuter(_linear.forward(audioFeatures));
    }
}

public class Qwen2AudioForConditionalGeneration : nn.Module, ICausalLM
{
    private readonly Qwen2AudioEncoder _audioTower;
    private readonly Qwen2AudioMultiModalProjector _multiModalProjector;
    private readonly Qwen2Model _languageModel;
    private readonly Linear _lmHead;
    private readonly Qwen2AudioConfig _config;

    public Qwen2AudioForConditionalGeneration(Qwen2AudioConfig config, string name = "Qwen2AudioForConditionalGeneration") : base(name)
    {
        _config = config;
        _audioTower = new Qwen2AudioEncoder(config.AudioConfig, name: "audio_tower");
        _multiModalProjector = new Qwen2AudioMultiModalProjector(config, name: "multi_modal_projector");
        
        _languageModel = new Qwen2Model(config.TextConfig, name: "language_model.model");
        _lmHead = Linear(config.TextConfig.HiddenSize, config.TextConfig.VocabSize, hasBias: false);

        register_module("audio_tower", _audioTower);
        register_module("multi_modal_projector", _multiModalProjector);
        register_module("language_model.model", _languageModel);
        register_module("language_model.lm_head", _lmHead);
    }

    public Tensor forward(Tensor inputIds, Tensor? positionIds = null, KVCache? kvCache = null)
    {
        // Default text-only forward supporting the ICausalLM interface.
        using var scope = torch.NewDisposeScope();
        var hiddenStates = _languageModel.forward(inputIds, positionIds, kvCache);
        var logits = _lmHead.forward(hiddenStates);
        return scope.MoveToOuter(logits);
    }

    /// <summary>
    /// Multimodal forward. Replaces special audio tokens in `inputIds` with projected audio features.
    /// Note: The inputIds must already be padded with enough tokens corresponding to the extracted audio features.
    /// </summary>
    public Tensor forward(Tensor inputIds, Tensor inputFeatures, Tensor? positionIds = null, KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();

        // 1. Get raw text embeddings
        var inputsEmbeds = _languageModel.GetInputEmbeddings().forward(inputIds);

        // 2. Extract audio features
        var audioOutputs = _audioTower.forward(inputFeatures); // [batch_size, audio_seq_length, audio_d_model]
        var audioFeaturesProjected = _multiModalProjector.forward(audioOutputs); // [batch_size, audio_seq_length, text_hidden_size]

        // 3. Find audio tokens in inputIds and inject the features
        var audioTokenMask = inputIds == _config.AudioTokenIndex;
        var audioTokenCount = audioTokenMask.sum().item<long>();
        var expectedCount = audioFeaturesProjected.shape[0] * audioFeaturesProjected.shape[1];

        if (audioTokenCount > 0 && audioTokenCount == expectedCount)
        {
            inputsEmbeds[audioTokenMask] = audioFeaturesProjected.view(-1, audioFeaturesProjected.shape[^1]).to_type(inputsEmbeds.dtype);
        }
        else if (audioTokenCount > 0)
        {
            System.Console.WriteLine($"Warning: Audio token count in input ({audioTokenCount}) doesn't match expected ({expectedCount}). Proceeding without replacing.");
        }

        // 4. Pass merged embeddings to language model
        var hiddenStates = _languageModel.forward(inputIds: null, positionIds: positionIds, kvCache: kvCache, inputsEmbeds: inputsEmbeds);
        var logits = _lmHead.forward(hiddenStates);

        return scope.MoveToOuter(logits);
    }

    public static Qwen2AudioForConditionalGeneration FromPretrained(string modelDirectory)
    {
        var configPath = System.IO.Path.Combine(modelDirectory, "config.json");
        var configJson = System.IO.File.ReadAllText(configPath);
        var config = System.Text.Json.JsonSerializer.Deserialize<Qwen2AudioConfig>(configJson) 
                     ?? throw new System.InvalidOperationException("Failed to load config.json");

        var model = new Qwen2AudioForConditionalGeneration(config);
        
        var stateDict = SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        model.load_state_dict(stateDict, strict: false);
        
        if (config.TextConfig.TieWordEmbeddings)
        {
            var embedWeight = model.get_parameter("language_model.model.embed_tokens.weight");
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
