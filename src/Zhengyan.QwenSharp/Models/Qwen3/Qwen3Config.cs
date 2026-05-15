using System.Text.Json.Serialization;
using Zhengyan.QwenSharp.Core;

namespace Zhengyan.QwenSharp.Models.Qwen3;

/// <summary>
/// Configuration for the Qwen3 model.
/// </summary>
public class Qwen3Config : ModelConfig
{
    [JsonPropertyName("vocab_size")]
    public new int VocabSize { get; set; } = 151936;

    [JsonPropertyName("hidden_size")]
    public new long HiddenSize { get; set; } = 4096;

    [JsonPropertyName("intermediate_size")]
    public new long IntermediateSize { get; set; } = 22016;

    [JsonPropertyName("num_hidden_layers")]
    public new int NumHiddenLayers { get; set; } = 32;

    [JsonPropertyName("num_attention_heads")]
    public new int NumAttentionHeads { get; set; } = 32;

    [JsonPropertyName("num_key_value_heads")]
    public new int NumKeyValueHeads { get; set; } = 32;

    [JsonPropertyName("head_dim")]
    public int HeadDim { get; set; } = 128;

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
}
