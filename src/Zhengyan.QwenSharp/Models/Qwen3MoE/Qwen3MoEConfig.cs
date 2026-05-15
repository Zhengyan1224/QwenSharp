using System.Text.Json.Serialization;
using Zhengyan.QwenSharp.Core;

namespace Zhengyan.QwenSharp.Models.Qwen3MoE;

public class Qwen3MoEConfig : ModelConfig
{
    [JsonPropertyName("vocab_size")]
    public new int VocabSize { get; set; } = 151936;

    [JsonPropertyName("hidden_size")]
    public new long HiddenSize { get; set; } = 2048;

    [JsonPropertyName("intermediate_size")]
    public new long IntermediateSize { get; set; } = 6144;

    [JsonPropertyName("num_hidden_layers")]
    public new int NumHiddenLayers { get; set; } = 24;

    [JsonPropertyName("num_attention_heads")]
    public new int NumAttentionHeads { get; set; } = 32;

    [JsonPropertyName("num_key_value_heads")]
    public new int NumKeyValueHeads { get; set; } = 4;

    [JsonPropertyName("head_dim")]
    public int HeadDim { get; set; } = 128; // fallback to hidden_size / num_attention_heads normally

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

    [JsonPropertyName("decoder_sparse_step")]
    public int DecoderSparseStep { get; set; } = 1;

    [JsonPropertyName("moe_intermediate_size")]
    public long MoeIntermediateSize { get; set; } = 768;

    [JsonPropertyName("num_experts_per_tok")]
    public int NumExpertsPerTok { get; set; } = 8;

    [JsonPropertyName("num_experts")]
    public int NumExperts { get; set; } = 128;

    [JsonPropertyName("norm_topk_prob")]
    public bool NormTopkProb { get; set; } = false;

    [JsonPropertyName("output_router_logits")]
    public bool OutputRouterLogits { get; set; } = false;

    [JsonPropertyName("router_aux_loss_coef")]
    public double RouterAuxLossCoef { get; set; } = 0.001;

    [JsonPropertyName("mlp_only_layers")]
    public int[] MlpOnlyLayers { get; set; } = System.Array.Empty<int>();
}
