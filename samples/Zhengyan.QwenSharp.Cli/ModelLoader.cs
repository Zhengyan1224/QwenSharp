using System;
using System.IO;
using System.Text.Json;
using Zhengyan.QwenSharp.Models;
using Zhengyan.QwenSharp.Models.Qwen2;
using Zhengyan.QwenSharp.Models.Qwen2MoE;
using Zhengyan.QwenSharp.Models.Qwen3;
using Zhengyan.QwenSharp.Models.Qwen3MoE;
using Zhengyan.QwenSharp.Models.Qwen35;
using Zhengyan.QwenSharp.Models.Qwen35MoE;
using Zhengyan.QwenSharp.Models.Qwen35VL;
using Zhengyan.QwenSharp.Models.Qwen2Audio;
using Zhengyan.QwenSharp.Models.Qwen25Omni;
using Zhengyan.QwenSharp.Models.Qwen3VL;

namespace Zhengyan.QwenSharp.Cli;

public static class ModelLoader
{
    public static ICausalLM LoadModel(string modelPath, out string modelType)
    {
        var configPath = Path.Combine(modelPath, "config.json");
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"Model config not found at {configPath}");
        }

        var json = File.ReadAllText(configPath);
        using var doc = JsonDocument.Parse(json);
        
        if (!doc.RootElement.TryGetProperty("model_type", out var typeProp))
        {
            throw new InvalidOperationException("config.json missing 'model_type' property.");
        }

        modelType = typeProp.GetString() ?? "";
        var hasVisionConfig = doc.RootElement.TryGetProperty("vision_config", out _);

        // Common Qwen types mapping. Note: adjust cases based on actual HF model types
        return modelType.ToLowerInvariant() switch
        {
            "qwen2" => Qwen2ForCausalLM.FromPretrained(modelPath),
            "qwen2_moe" => Qwen2MoEForCausalLM.FromPretrained(modelPath),
            "qwen3" => Qwen3ForCausalLM.FromPretrained(modelPath),
            "qwen3_moe" => Qwen3MoEForCausalLM.FromPretrained(modelPath),
            "qwen35" or "qwen3.5" or "qwen3_5" or "qwen3_5_text" => hasVisionConfig
                ? Qwen35VLForConditionalGeneration.FromPretrained(modelPath)
                : Qwen35ForCausalLM.FromPretrained(modelPath),
            "qwen35_moe" or "qwen3.5_moe" or "qwen3_5_moe" or "qwen3_5_text_moe" => Qwen35MoEForCausalLM.FromPretrained(modelPath),
            "qwen3_vl" => Qwen3VLForConditionalGeneration.FromPretrained(modelPath),
            "qwen2_audio" => Qwen2AudioForConditionalGeneration.FromPretrained(modelPath),
            "qwen2_5_omni" => Qwen25OmniForConditionalGeneration.FromPretrained(modelPath),
            _ => Qwen2ForCausalLM.FromPretrained(modelPath) // Default fallback
        };
    }

    public static IMultimodalCausalLM LoadVisionLanguageModel(string modelPath, out string modelType)
    {
        var json = File.ReadAllText(Path.Combine(modelPath, "config.json"));
        using var doc = JsonDocument.Parse(json);
        modelType = doc.RootElement.GetProperty("model_type").GetString() ?? "";

        return modelType.ToLowerInvariant() switch
        {
            "qwen35" or "qwen3.5" or "qwen3_5" or "qwen3_5_text" => Qwen35VLForConditionalGeneration.FromPretrained(modelPath),
            "qwen3_vl" => Qwen3VLForConditionalGeneration.FromPretrained(modelPath),
            _ => throw new NotSupportedException($"Model type '{modelType}' is not a supported vision-language model.")
        };
    }

    public static TorchSharp.torch.nn.Module LoadSequenceClassificationModel(string modelPath, out string modelType)
    {
        var json = File.ReadAllText(Path.Combine(modelPath, "config.json"));
        using var doc = JsonDocument.Parse(json);
        modelType = doc.RootElement.GetProperty("model_type").GetString() ?? "";
        return modelType.ToLowerInvariant() switch
        {
            "qwen2_moe" => Qwen2MoEForSequenceClassification.FromPretrained(modelPath),
            "qwen3" => Qwen3ForSequenceClassification.FromPretrained(modelPath),
            "qwen3_moe" => Qwen3MoEForSequenceClassification.FromPretrained(modelPath),
            "qwen35" or "qwen3.5" or "qwen3_5" or "qwen3_5_text" => Qwen35ForSequenceClassification.FromPretrained(modelPath),
            "qwen35_moe" or "qwen3.5_moe" or "qwen3_5_moe" or "qwen3_5_text_moe" => Qwen35MoEForSequenceClassification.FromPretrained(modelPath),
            _ => Qwen2ForSequenceClassification.FromPretrained(modelPath)
        };
    }

    public static TorchSharp.torch.nn.Module LoadTokenClassificationModel(string modelPath, out string modelType)
    {
        var json = File.ReadAllText(Path.Combine(modelPath, "config.json"));
        using var doc = JsonDocument.Parse(json);
        modelType = doc.RootElement.GetProperty("model_type").GetString() ?? "";
        return modelType.ToLowerInvariant() switch
        {
            "qwen2_moe" => Qwen2MoEForTokenClassification.FromPretrained(modelPath),
            "qwen3" => Qwen3ForTokenClassification.FromPretrained(modelPath),
            "qwen3_moe" => Qwen3MoEForTokenClassification.FromPretrained(modelPath),
            "qwen35" or "qwen3.5" or "qwen3_5" or "qwen3_5_text" => Qwen35ForTokenClassification.FromPretrained(modelPath),
            "qwen35_moe" or "qwen3.5_moe" or "qwen3_5_moe" or "qwen3_5_text_moe" => Qwen35MoEForTokenClassification.FromPretrained(modelPath),
            _ => Qwen2ForTokenClassification.FromPretrained(modelPath)
        };
    }

    public static TorchSharp.torch.nn.Module LoadQuestionAnsweringModel(string modelPath, out string modelType)
    {
        var json = File.ReadAllText(Path.Combine(modelPath, "config.json"));
        using var doc = JsonDocument.Parse(json);
        modelType = doc.RootElement.GetProperty("model_type").GetString() ?? "";
        return modelType.ToLowerInvariant() switch
        {
            "qwen2_moe" => Qwen2MoEForQuestionAnswering.FromPretrained(modelPath),
            "qwen3" => Qwen3ForQuestionAnswering.FromPretrained(modelPath),
            "qwen3_moe" => Qwen3MoEForQuestionAnswering.FromPretrained(modelPath),
            "qwen35" or "qwen3.5" or "qwen3_5" or "qwen3_5_text" => Qwen35ForQuestionAnswering.FromPretrained(modelPath),
            "qwen35_moe" or "qwen3.5_moe" or "qwen3_5_moe" or "qwen3_5_text_moe" => Qwen35MoEForQuestionAnswering.FromPretrained(modelPath),
            _ => Qwen2ForQuestionAnswering.FromPretrained(modelPath)
        };
    }

    public static TorchSharp.torch.nn.Module LoadRetrievalModel(string modelPath, out string modelType)
    {
        var json = File.ReadAllText(Path.Combine(modelPath, "config.json"));
        using var doc = JsonDocument.Parse(json);
        modelType = doc.RootElement.GetProperty("model_type").GetString() ?? "";
        return modelType.ToLowerInvariant() switch
        {
            "colqwen2" => Zhengyan.QwenSharp.Models.ColQwen2.ColQwen2ForRetrieval.FromPretrained(modelPath),
            _ => Zhengyan.QwenSharp.Models.ColQwen2.ColQwen2ForRetrieval.FromPretrained(modelPath)
        };
    }

    public static TorchSharp.torch.nn.Module LoadTalkerModel(string modelPath, out string modelType)
    {
        var json = File.ReadAllText(Path.Combine(modelPath, "config.json"));
        using var doc = JsonDocument.Parse(json);
        modelType = doc.RootElement.GetProperty("model_type").GetString() ?? "";
        return modelType.ToLowerInvariant() switch
        {
            "qwen2_5_omni_talker" => Zhengyan.QwenSharp.Models.Qwen25Omni.Qwen25OmniTalkerForConditionalGeneration.FromPretrained(modelPath),
            _ => Zhengyan.QwenSharp.Models.Qwen25Omni.Qwen25OmniTalkerForConditionalGeneration.FromPretrained(modelPath)
        };
    }

    public static TorchSharp.torch.nn.Module LoadToken2WavModel(string modelPath, out string modelType)
    {
        var json = File.ReadAllText(Path.Combine(modelPath, "config.json"));
        using var doc = JsonDocument.Parse(json);
        modelType = doc.RootElement.GetProperty("model_type").GetString() ?? "";
        return modelType.ToLowerInvariant() switch
        {
            "qwen2_5_omni_token2wav" => Zhengyan.QwenSharp.Models.Qwen25Omni.Qwen25OmniToken2WavModel.FromPretrained(modelPath),
            _ => throw new NotSupportedException($"Unsupported Token2Wav model type: {modelType}")
        };
    }
}
