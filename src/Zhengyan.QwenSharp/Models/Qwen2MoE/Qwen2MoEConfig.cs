using System.Text.Json.Serialization;
using Zhengyan.QwenSharp.Core;

namespace Zhengyan.QwenSharp.Models.Qwen2MoE;

/// <summary>
/// Configuration for the Qwen2-MoE model.
/// </summary>
public class Qwen2MoEConfig : ModelConfig
{
    [JsonPropertyName("vocab_size")]
    public new int VocabSize { get; set; } = 151936;

    [JsonPropertyName("hidden_size")]
    public new long HiddenSize { get; set; } = 2048;

    [JsonPropertyName("intermediate_size")]
    public new long IntermediateSize { get; set; } = 5632;

    [JsonPropertyName("num_hidden_layers")]
    public new int NumHiddenLayers { get; set; } = 24;

    [JsonPropertyName("num_attention_heads")]
    public new int NumAttentionHeads { get; set; } = 16;

    [JsonPropertyName("num_key_value_heads")]
    public new int NumKeyValueHeads { get; set; } = 16;

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

    // MoE specific arguments
    [JsonPropertyName("decoder_sparse_step")]
    public int DecoderSparseStep { get; set; } = 1;

    [JsonPropertyName("moe_intermediate_size")]
    public long MoeIntermediateSize { get; set; } = 1408;

    [JsonPropertyName("shared_expert_intermediate_size")]
    public long SharedExpertIntermediateSize { get; set; } = 5632;

    [JsonPropertyName("num_experts_per_tok")]
    public int NumExpertsPerTok { get; set; } = 4;

    [JsonPropertyName("num_experts")]
    public int NumExperts { get; set; } = 60;

    [JsonPropertyName("norm_topk_prob")]
    public bool NormTopkProb { get; set; } = false;

    [JsonPropertyName("output_router_logits")]
    public bool OutputRouterLogits { get; set; } = false;

    [JsonPropertyName("router_aux_loss_coef")]
    public double RouterAuxLossCoef { get; set; } = 0.001;

    [JsonPropertyName("mlp_only_layers")]
    public int[] MlpOnlyLayers { get; set; } = System.Array.Empty<int>();
}
