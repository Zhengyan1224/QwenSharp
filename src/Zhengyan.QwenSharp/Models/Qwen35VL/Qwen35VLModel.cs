using System;
using System.IO;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using Zhengyan.QwenSharp.Models.Qwen35;
using Zhengyan.QwenSharp.Models.Vision;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen35VL;

public class Qwen35VLModel : nn.Module
{
    private readonly Qwen35VLConfig _config;
    private long? _ropeDelta;
    public readonly Qwen35VisionTransformer Visual;
    public readonly Qwen35Model LanguageModel;

    public Qwen35VLModel(Qwen35VLConfig config, string name = "model") : base(name)
    {
        _config = config;
        Visual = new Qwen35VisionTransformer(config.VisionConfig, name: "visual");
        LanguageModel = new Qwen35Model(config.TextConfig, name: "language_model");

        register_module("visual", Visual);
        register_module("language_model", LanguageModel);
    }

    private Tensor BuildMmTokenTypeIds(Tensor inputIds)
    {
        using var scope = torch.NewDisposeScope();
        var mmTokenTypeIds = zeros_like(inputIds, dtype: ScalarType.Int64);
        mmTokenTypeIds = where(inputIds == _config.ImageTokenId, ones_like(mmTokenTypeIds), mmTokenTypeIds);
        return scope.MoveToOuter(mmTokenTypeIds);
    }

    private Tensor GetPlaceholderMask(Tensor inputIds, Tensor inputsEmbeds, Tensor imageEmbeds)
    {
        using var scope = torch.NewDisposeScope();
        var imageMask = inputIds == _config.ImageTokenId;
        var expandedMask = imageMask.unsqueeze(-1).expand_as(inputsEmbeds).to(inputsEmbeds.device);
        long imageTokenCount = imageMask.sum().item<long>();
        long maskedElements = expandedMask.sum().item<long>();
        long featureElements = imageEmbeds.numel();

        if (maskedElements != featureElements)
        {
            throw new InvalidOperationException($"Image features and image tokens do not match, tokens: {imageTokenCount}, features: {imageEmbeds.shape[0]}");
        }

        return scope.MoveToOuter(expandedMask);
    }

    private Tensor ComputePositionIds(
        Tensor inputIds,
        Tensor inputsEmbeds,
        Tensor mmTokenTypeIds,
        Tensor attentionMask,
        KVCache? kvCache,
        Tensor? imageGridThw)
    {
        using var scope = torch.NewDisposeScope();

        if (imageGridThw is not null && (kvCache is null || kvCache.GetSeqLength() == 0))
        {
            var promptPositionIds = QwenVLPositionHelper.BuildPromptPositionIds(
                inputIds,
                mmTokenTypeIds,
                _config.VisionConfig.SpatialMergeSize,
                imageGridThw,
                out var ropeDelta,
                attentionMask);
            _ropeDelta = ropeDelta;
            return scope.MoveToOuter(promptPositionIds);
        }

        if (_ropeDelta.HasValue)
        {
            return scope.MoveToOuter(QwenVLPositionHelper.BuildDecodePositionIds(inputIds, kvCache, _ropeDelta.Value));
        }

        var batchSize = inputIds.shape[0];
        var seqLen = inputIds.shape[1];
        var pastLength = kvCache?.GetSeqLength() ?? 0;
        var fallbackPositionIds = torch.arange(pastLength, pastLength + seqLen, device: inputIds.device, dtype: ScalarType.Int64)
            .view(1, 1, seqLen)
            .expand(3, batchSize, seqLen)
            .clone();
        return scope.MoveToOuter(fallbackPositionIds);
    }

    public Tensor forward(
        Tensor inputIds,
        Tensor? positionIds = null,
        KVCache? kvCache = null,
        Tensor? pixelValues = null,
        Tensor? imageGridThw = null)
    {
        using var scope = torch.NewDisposeScope();

        var inputsEmbeds = LanguageModel.EmbedTokens(inputIds);
        using var attentionMask = ones_like(inputIds, dtype: ScalarType.Int64);
        using var mmTokenTypeIds = BuildMmTokenTypeIds(inputIds);

        if (pixelValues is not null && imageGridThw is not null)
        {
            using var imageEmbeds = Visual.forward(pixelValues, imageGridThw).to(inputsEmbeds.device).to_type(inputsEmbeds.dtype);
            using var imageMask = GetPlaceholderMask(inputIds, inputsEmbeds, imageEmbeds);
            inputsEmbeds = inputsEmbeds.masked_scatter(imageMask, imageEmbeds);
        }

        positionIds ??= ComputePositionIds(inputIds, inputsEmbeds, mmTokenTypeIds, attentionMask, kvCache, imageGridThw);
        var outputs = LanguageModel.ForwardEmbeddings(inputsEmbeds, positionIds, kvCache);
        return scope.MoveToOuter(outputs);
    }
}

public class Qwen35VLForConditionalGeneration : nn.Module, IMultimodalCausalLM
{
    private readonly Qwen35VLConfig _config;
    public readonly Qwen35VLModel Model;
    public readonly Linear LmHead;

    public Qwen35VLForConditionalGeneration(Qwen35VLConfig config, string name = "model") : base(name)
    {
        _config = config;
        Model = new Qwen35VLModel(config, name: "model");
        LmHead = Linear(config.TextConfig.HiddenSize, config.TextConfig.VocabSize, hasBias: false);

        register_module("model", Model);
        register_module("lm_head", LmHead);
    }

    public Tensor forward(
        Tensor inputIds,
        Tensor? positionIds = null,
        KVCache? kvCache = null,
        Tensor? pixelValues = null,
        Tensor? imageGridThw = null)
    {
        using var scope = torch.NewDisposeScope();
        var hiddenStates = Model.forward(inputIds, positionIds, kvCache, pixelValues, imageGridThw);
        return scope.MoveToOuter(LmHead.forward(hiddenStates));
    }

    Tensor ICausalLM.forward(Tensor inputIds, Tensor? positionIds, KVCache? kvCache)
        => forward(inputIds, positionIds, kvCache, null, null);

    public static Qwen35VLForConditionalGeneration FromPretrained(string modelDirectory)
    {
        var config = LoadConfig(modelDirectory);
        var model = new Qwen35VLForConditionalGeneration(config);
        var stateDict = Zhengyan.QwenSharp.Core.SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        model.load_state_dict(stateDict, strict: false);

        if (config.TextConfig.TieWordEmbeddings)
        {
            var embedWeight = model.get_parameter("model.language_model.embed_tokens.weight");
            var lmHeadWeight = model.LmHead.get_parameter("weight");
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

    public static Qwen35VLConfig LoadConfig(string modelDirectory)
    {
        var configPath = Path.Combine(modelDirectory, "config.json");
        var configJson = File.ReadAllText(configPath);
        var config = JsonSerializer.Deserialize<Qwen35VLConfig>(configJson)
                     ?? throw new InvalidOperationException("Failed to load Qwen3.5 VL config.");

        Qwen35ForCausalLM.PopulateExtendedConfig(config.TextConfig);
        return config;
    }
}
