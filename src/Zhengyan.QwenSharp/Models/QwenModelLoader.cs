using System.Text.Json;
using Zhengyan.QwenSharp.Models.Qwen2;
using Zhengyan.QwenSharp.Models.Qwen2Audio;
using Zhengyan.QwenSharp.Models.Qwen2MoE;
using Zhengyan.QwenSharp.Models.Qwen25Omni;
using Zhengyan.QwenSharp.Models.Qwen35;
using Zhengyan.QwenSharp.Models.Qwen35MoE;
using Zhengyan.QwenSharp.Models.Qwen35VL;
using Zhengyan.QwenSharp.Models.Qwen3;
using Zhengyan.QwenSharp.Models.Qwen3MoE;
using Zhengyan.QwenSharp.Models.Qwen3VL;

namespace Zhengyan.QwenSharp.Models;

public static class QwenModelLoader
{
    public static ICausalLM LoadCausalLM(string modelPath, out string modelType)
    {
        using var document = ReadConfig(modelPath);
        modelType = ReadModelType(document);
        var hasVisionConfig = document.RootElement.TryGetProperty("vision_config", out _);

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
            _ => Qwen2ForCausalLM.FromPretrained(modelPath),
        };
    }

    public static string ReadModelType(string modelPath)
    {
        using var document = ReadConfig(modelPath);
        return ReadModelType(document);
    }

    public static bool IsQwen25Omni(string modelPath)
        => string.Equals(ReadModelType(modelPath), "qwen2_5_omni", StringComparison.OrdinalIgnoreCase);

    private static JsonDocument ReadConfig(string modelPath)
    {
        var configPath = Path.Combine(modelPath, "config.json");
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"Model config not found at {configPath}", configPath);
        }

        return JsonDocument.Parse(File.ReadAllText(configPath));
    }

    private static string ReadModelType(JsonDocument document)
    {
        if (!document.RootElement.TryGetProperty("model_type", out var typeProperty))
        {
            throw new InvalidOperationException("config.json missing 'model_type' property.");
        }

        return typeProperty.GetString() ?? string.Empty;
    }
}
