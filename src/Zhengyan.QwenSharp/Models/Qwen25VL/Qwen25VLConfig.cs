using System.Linq;
using System.Text.Json.Serialization;
using Zhengyan.QwenSharp.Core;

namespace Zhengyan.QwenSharp.Models.Qwen25VL;

public class Qwen25VLVisionConfig
{
    [JsonPropertyName("depth")]
    public int Depth { get; set; } = 32;

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; } = 3584;

    [JsonPropertyName("hidden_act")]
    public string HiddenAct { get; set; } = "silu";

    [JsonPropertyName("intermediate_size")]
    public int IntermediateSize { get; set; } = 3420;

    [JsonPropertyName("num_heads")]
    public int NumHeads { get; set; } = 16;

    [JsonPropertyName("in_channels")]
    public int InChannels { get; set; } = 3;

    [JsonPropertyName("patch_size")]
    public int PatchSize { get; set; } = 14;

    [JsonPropertyName("spatial_merge_size")]
    public int SpatialMergeSize { get; set; } = 2;

    [JsonPropertyName("temporal_patch_size")]
    public int TemporalPatchSize { get; set; } = 2;

    [JsonPropertyName("tokens_per_second")]
    public int TokensPerSecond { get; set; } = 4;

    [JsonPropertyName("window_size")]
    public int WindowSize { get; set; } = 112;

    [JsonPropertyName("out_hidden_size")]
    public int OutHiddenSize { get; set; } = 3584;

    [JsonPropertyName("fullatt_block_indexes")]
    public int[] FullattBlockIndexes { get; set; } = new[] { 7, 15, 23, 31 };

    [JsonPropertyName("initializer_range")]
    public double InitializerRange { get; set; } = 0.02;
}

public class Qwen25VLTextConfig : ModelConfig
{
    [JsonPropertyName("vocab_size")]
    public new int VocabSize { get; set; } = 152064;

    [JsonPropertyName("max_position_embeddings")]
    public new int MaxPositionEmbeddings { get; set; } = 32768;

    [JsonPropertyName("hidden_size")]
    public new int HiddenSize { get; set; } = 8192;

    [JsonPropertyName("intermediate_size")]
    public new int IntermediateSize { get; set; } = 29568;

    [JsonPropertyName("num_hidden_layers")]
    public new int NumHiddenLayers { get; set; } = 80;

    [JsonPropertyName("num_attention_heads")]
    public new int NumAttentionHeads { get; set; } = 64;

    [JsonPropertyName("num_key_value_heads")]
    public new int NumKeyValueHeads { get; set; } = 8;

    [JsonPropertyName("hidden_act")]
    public new string HiddenAct { get; set; } = "silu";

    [JsonPropertyName("initializer_range")]
    public new double InitializerRange { get; set; } = 0.02;

    [JsonPropertyName("rms_norm_eps")]
    public new double RmsNormEps { get; set; } = 1e-05;

    [JsonPropertyName("use_cache")]
    public new bool UseCache { get; set; } = true;

    [JsonPropertyName("use_sliding_window")]
    public bool UseSlidingWindow { get; set; } = false;

    [JsonPropertyName("sliding_window")]
    public int? SlidingWindow { get; set; } = 4096;

    [JsonPropertyName("max_window_layers")]
    public int MaxWindowLayers { get; set; } = 80;

    [JsonPropertyName("attention_dropout")]
    public new double AttentionDropout { get; set; } = 0.0;

    [JsonPropertyName("layer_types")]
    public string[] LayerTypes { get; set; }

    [JsonPropertyName("rope_scaling")]
    public System.Collections.Generic.Dictionary<string, object> RopeScaling { get; set; }

    public int[] MRopeSection 
    {
        get 
        {
            if (RopeScaling != null && RopeScaling.TryGetValue("mrope_section", out var val) && val is System.Text.Json.JsonElement element && element.ValueKind == System.Text.Json.JsonValueKind.Array)
            {
                return element.EnumerateArray().Select(x => x.GetInt32()).ToArray();
            }
            return new int[] { 16, 24, 24 }; // Fallback for 128 head dim
        }
    }
}

public class Qwen25VLConfig : ModelConfig
{
    [JsonPropertyName("vision_config")]
    public Qwen25VLVisionConfig VisionConfig { get; set; }

    [JsonPropertyName("text_config")]
    public Qwen25VLTextConfig TextConfig { get; set; }

    [JsonPropertyName("image_token_id")]
    public int ImageTokenId { get; set; } = 151655;

    [JsonPropertyName("video_token_id")]
    public int VideoTokenId { get; set; } = 151656;

    [JsonPropertyName("vision_start_token_id")]
    public int VisionStartTokenId { get; set; } = 151652;

    [JsonPropertyName("vision_end_token_id")]
    public int VisionEndTokenId { get; set; } = 151653;
}
