using System.Linq;
using System.Text.Json.Serialization;
using Zhengyan.QwenSharp.Core;

namespace Zhengyan.QwenSharp.Models.Qwen2VL;

public class Qwen2VLVisionConfig
{
    [JsonPropertyName("depth")]
    public int Depth { get; set; } = 32;

    [JsonPropertyName("embed_dim")]
    public int EmbedDim { get; set; } = 1280;

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; } = 3584;

    [JsonPropertyName("hidden_act")]
    public string HiddenAct { get; set; } = "quick_gelu";

    [JsonPropertyName("mlp_ratio")]
    public double MlpRatio { get; set; } = 4;

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
}

public class Qwen2VLConfig : ModelConfig
{
    [JsonPropertyName("vision_config")]
    public Qwen2VLVisionConfig VisionConfig { get; set; } = new();

    // LLM Config
    [JsonPropertyName("vocab_size")]
    public new int VocabSize { get; set; } = 152064;

    [JsonPropertyName("hidden_size")]
    public new long HiddenSize { get; set; } = 8192;

    [JsonPropertyName("intermediate_size")]
    public new long IntermediateSize { get; set; } = 29568;

    [JsonPropertyName("num_hidden_layers")]
    public new int NumHiddenLayers { get; set; } = 80;

    [JsonPropertyName("num_attention_heads")]
    public new int NumAttentionHeads { get; set; } = 64;

    [JsonPropertyName("num_key_value_heads")]
    public new int NumKeyValueHeads { get; set; } = 8;

    [JsonPropertyName("rms_norm_eps")]
    public new double RmsNormEps { get; set; } = 1e-05;

    [JsonPropertyName("max_position_embeddings")]
    public new int MaxPositionEmbeddings { get; set; } = 32768;

    [JsonPropertyName("attention_dropout")]
    public new double AttentionDropout { get; set; } = 0.0;
    
    [JsonPropertyName("rope_theta")]
    public new double RopeTheta { get; set; } = 1000000.0;
    
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
    
    [JsonPropertyName("image_token_id")]
    public int ImageTokenId { get; set; } = 151655;

    [JsonPropertyName("video_token_id")]
    public int VideoTokenId { get; set; } = 151656;

    [JsonPropertyName("vision_start_token_id")]
    public int VisionStartTokenId { get; set; } = 151652;

    [JsonPropertyName("vision_end_token_id")]
    public int VisionEndTokenId { get; set; } = 151653;
}
