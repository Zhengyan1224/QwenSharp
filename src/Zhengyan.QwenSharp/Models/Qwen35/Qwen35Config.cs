using System.Text.Json.Serialization;
using System.Text.Json;
using Zhengyan.QwenSharp.Core;

namespace Zhengyan.QwenSharp.Models.Qwen35;

public class Qwen35Config : ModelConfig
{
    [JsonPropertyName("vocab_size")]
    public new int VocabSize { get; set; } = 248320;

    [JsonPropertyName("hidden_size")]
    public new long HiddenSize { get; set; } = 4096;

    [JsonPropertyName("intermediate_size")]
    public new long IntermediateSize { get; set; } = 12288;

    [JsonPropertyName("num_hidden_layers")]
    public new int NumHiddenLayers { get; set; } = 32;

    [JsonPropertyName("num_attention_heads")]
    public new int NumAttentionHeads { get; set; } = 16;

    [JsonPropertyName("num_key_value_heads")]
    public new int NumKeyValueHeads { get; set; } = 4;

    [JsonPropertyName("head_dim")]
    public int HeadDim { get; set; } = 256;

    [JsonPropertyName("rms_norm_eps")]
    public new double RmsNormEps { get; set; } = 1e-6;

    [JsonPropertyName("rope_theta")]
    public new double RopeTheta { get; set; } = 10000.0;

    [JsonPropertyName("max_position_embeddings")]
    public new int MaxPositionEmbeddings { get; set; } = 32768;

    [JsonPropertyName("attention_dropout")]
    public new double AttentionDropout { get; set; } = 0.0;

    [JsonPropertyName("tie_word_embeddings")]
    public new bool TieWordEmbeddings { get; set; } = false;

    [JsonPropertyName("attention_bias")]
    public bool AttentionBias { get; set; } = false;

    // Linear Attention Params
    [JsonPropertyName("linear_conv_kernel_dim")]
    public int LinearConvKernelDim { get; set; } = 4;

    [JsonPropertyName("linear_key_head_dim")]
    public int LinearKeyHeadDim { get; set; } = 128;

    [JsonPropertyName("linear_value_head_dim")]
    public int LinearValueHeadDim { get; set; } = 128;

    [JsonPropertyName("linear_num_key_heads")]
    public int LinearNumKeyHeads { get; set; } = 16;

    [JsonPropertyName("linear_num_value_heads")]
    public int LinearNumValueHeads { get; set; } = 32;

    [JsonPropertyName("layer_types")]
    public string[]? LayerTypes { get; set; }

    [JsonPropertyName("partial_rotary_factor")]
    public double PartialRotaryFactor { get; set; } = 0.25;

    [JsonPropertyName("rope_parameters")]
    public JsonElement? RopeParameters { get; set; }

    public int[] MRopeSection { get; set; } = [11, 11, 10];
}
