using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Zhengyan.QwenSharp.Core;

/// <summary>
/// Base class for all HuggingFace model configurations.
/// Deserializes from config.json using System.Text.Json.
/// </summary>
public class ModelConfig
{
    [JsonPropertyName("model_type")]
    public string? ModelType { get; set; }

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; } = 4096;

    [JsonPropertyName("num_hidden_layers")]
    public int NumHiddenLayers { get; set; } = 32;

    [JsonPropertyName("num_attention_heads")]
    public int NumAttentionHeads { get; set; } = 32;

    [JsonPropertyName("num_key_value_heads")]
    public int? NumKeyValueHeads { get; set; }

    [JsonPropertyName("intermediate_size")]
    public int IntermediateSize { get; set; } = 22016;

    [JsonPropertyName("hidden_act")]
    public string HiddenAct { get; set; } = "silu";

    [JsonPropertyName("max_position_embeddings")]
    public int MaxPositionEmbeddings { get; set; } = 131072;

    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; } = 151936;

    [JsonPropertyName("rms_norm_eps")]
    public float RmsNormEps { get; set; } = 1e-6f;

    [JsonPropertyName("rope_theta")]
    public float RopeTheta { get; set; } = 10000.0f;

    [JsonPropertyName("use_sliding_window")]
    public bool UseSlidingWindow { get; set; } = false;

    [JsonPropertyName("sliding_window")]
    public int? SlidingWindow { get; set; }

    [JsonPropertyName("max_window_layers")]
    public int MaxWindowLayers { get; set; } = 28;

    [JsonPropertyName("attention_dropout")]
    public float AttentionDropout { get; set; } = 0.0f;

    [JsonPropertyName("classifier_dropout")]
    public float? ClassifierDropout { get; set; }

    [JsonPropertyName("num_labels")]
    public int NumLabels { get; set; } = 2;

    [JsonPropertyName("tie_word_embeddings")]
    public bool TieWordEmbeddings { get; set; } = false;

    [JsonPropertyName("bos_token_id")]
    public int? BosTokenId { get; set; }

    [JsonPropertyName("eos_token_id")]
    [JsonConverter(typeof(IntOrIntArrayConverter))]
    public int[]? EosTokenId { get; set; }

    [JsonPropertyName("pad_token_id")]
    public int? PadTokenId { get; set; }

    [JsonPropertyName("torch_dtype")]
    public string? TorchDtype { get; set; }

    // Rope parameters (nested dict in newer models)
    [JsonPropertyName("rope_scaling")]
    public JsonElement? RopeScaling { get; set; }

    /// <summary>
    /// Gets the effective num_key_value_heads, falling back to num_attention_heads for MHA.
    /// </summary>
    public int EffectiveNumKeyValueHeads => NumKeyValueHeads ?? NumAttentionHeads;

    /// <summary>
    /// Loads config from a model directory's config.json file.
    /// </summary>
    public static T FromDirectory<T>(string modelDirectory) where T : ModelConfig
    {
        var configPath = Path.Combine(modelDirectory, "config.json");
        if (!File.Exists(configPath))
            throw new FileNotFoundException($"config.json not found in {modelDirectory}");

        var json = File.ReadAllText(configPath);
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
        };
        var config = JsonSerializer.Deserialize<T>(json, options)
            ?? throw new InvalidOperationException("Failed to deserialize config.json");
        return config;
    }

    /// <summary>
    /// Loads config from a JSON string.
    /// </summary>
    public static T FromJson<T>(string json) where T : ModelConfig
    {
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
        };
        return JsonSerializer.Deserialize<T>(json, options)
            ?? throw new InvalidOperationException("Failed to deserialize config JSON");
    }
}

/// <summary>
/// Converter that accepts either a single int or an array of int for eos_token_id.
/// </summary>
public class IntOrIntArrayConverter : JsonConverter<int[]?>
{
    public override int[]? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.Number)
        {
            return [reader.GetInt32()];
        }
        if (reader.TokenType == JsonTokenType.StartArray)
        {
            var list = new List<int>();
            while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
            {
                if (reader.TokenType == JsonTokenType.Number)
                    list.Add(reader.GetInt32());
            }
            return [.. list];
        }
        if (reader.TokenType == JsonTokenType.Null)
            return null;

        throw new JsonException($"Unexpected token {reader.TokenType} for eos_token_id");
    }

    public override void Write(Utf8JsonWriter writer, int[]? value, JsonSerializerOptions options)
    {
        if (value == null) { writer.WriteNullValue(); return; }
        if (value.Length == 1) { writer.WriteNumberValue(value[0]); return; }
        writer.WriteStartArray();
        foreach (var v in value) writer.WriteNumberValue(v);
        writer.WriteEndArray();
    }
}
