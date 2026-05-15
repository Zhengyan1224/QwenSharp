using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using Zhengyan.QwenSharp.Core;

namespace Zhengyan.QwenSharp.Models.Qwen3VL;

public class Qwen3VLVisionConfig
{
    [JsonPropertyName("depth")]
    public int Depth { get; set; } = 24;

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; } = 1024;

    [JsonPropertyName("hidden_act")]
    public string HiddenAct { get; set; } = "gelu_pytorch_tanh";

    [JsonPropertyName("intermediate_size")]
    public int IntermediateSize { get; set; } = 4096;

    [JsonPropertyName("num_heads")]
    public int NumHeads { get; set; } = 16;

    [JsonPropertyName("in_channels")]
    public int InChannels { get; set; } = 3;

    [JsonPropertyName("patch_size")]
    public int PatchSize { get; set; } = 16;

    [JsonPropertyName("spatial_merge_size")]
    public int SpatialMergeSize { get; set; } = 2;

    [JsonPropertyName("temporal_patch_size")]
    public int TemporalPatchSize { get; set; } = 2;

    [JsonPropertyName("out_hidden_size")]
    public int OutHiddenSize { get; set; } = 2048;

    [JsonPropertyName("num_position_embeddings")]
    public int NumPositionEmbeddings { get; set; } = 2304;

    [JsonPropertyName("deepstack_visual_indexes")]
    public int[] DeepstackVisualIndexes { get; set; } = [];
}

public class Qwen3VLTextConfig : ModelConfig
{
    [JsonPropertyName("vocab_size")]
    public new int VocabSize { get; set; } = 151936;

    [JsonPropertyName("max_position_embeddings")]
    public new int MaxPositionEmbeddings { get; set; } = 262144;

    [JsonPropertyName("hidden_size")]
    public new int HiddenSize { get; set; } = 2048;

    [JsonPropertyName("intermediate_size")]
    public new int IntermediateSize { get; set; } = 6144;

    [JsonPropertyName("num_hidden_layers")]
    public new int NumHiddenLayers { get; set; } = 28;

    [JsonPropertyName("num_attention_heads")]
    public new int NumAttentionHeads { get; set; } = 16;

    [JsonPropertyName("num_key_value_heads")]
    public new int NumKeyValueHeads { get; set; } = 8;

    [JsonPropertyName("head_dim")]
    public int HeadDim { get; set; } = 128;

    [JsonPropertyName("hidden_act")]
    public new string HiddenAct { get; set; } = "silu";

    [JsonPropertyName("initializer_range")]
    public new double InitializerRange { get; set; } = 0.02;

    [JsonPropertyName("rms_norm_eps")]
    public new double RmsNormEps { get; set; } = 1e-6;

    [JsonPropertyName("use_cache")]
    public new bool UseCache { get; set; } = true;

    [JsonPropertyName("attention_dropout")]
    public new double AttentionDropout { get; set; } = 0.0;

    [JsonPropertyName("rope_theta")]
    public new double RopeTheta { get; set; } = 5000000.0;

    [JsonPropertyName("tie_word_embeddings")]
    public new bool TieWordEmbeddings { get; set; } = true;

    [JsonPropertyName("rope_scaling")]
    public Dictionary<string, object>? RopeScaling { get; set; }

    public int[] MRopeSection
    {
        get
        {
            if (RopeScaling != null &&
                RopeScaling.TryGetValue("mrope_section", out var val) &&
                val is JsonElement element &&
                element.ValueKind == JsonValueKind.Array)
            {
                return element.EnumerateArray().Select(x => x.GetInt32()).ToArray();
            }

            return [24, 20, 20];
        }
    }
}

public class Qwen3VLConfig : ModelConfig
{
    [JsonPropertyName("vision_config")]
    public Qwen3VLVisionConfig VisionConfig { get; set; } = new();

    [JsonPropertyName("text_config")]
    public Qwen3VLTextConfig TextConfig { get; set; } = new();

    [JsonPropertyName("image_token_id")]
    public int ImageTokenId { get; set; } = 151655;

    [JsonPropertyName("video_token_id")]
    public int VideoTokenId { get; set; } = 151656;

    [JsonPropertyName("vision_start_token_id")]
    public int VisionStartTokenId { get; set; } = 151652;

    [JsonPropertyName("vision_end_token_id")]
    public int VisionEndTokenId { get; set; } = 151653;
}

