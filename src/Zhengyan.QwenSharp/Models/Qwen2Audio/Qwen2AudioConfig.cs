using System.Text.Json.Serialization;
using Zhengyan.QwenSharp.Models.Qwen2;

namespace Zhengyan.QwenSharp.Models.Qwen2Audio;

public class Qwen2AudioEncoderConfig
{
    [JsonPropertyName("num_mel_bins")]
    public int NumMelBins { get; set; } = 128;

    [JsonPropertyName("d_model")]
    public int DModel { get; set; } = 1024;

    [JsonPropertyName("encoder_layers")]
    public int EncoderLayers { get; set; } = 32;

    [JsonPropertyName("encoder_attention_heads")]
    public int EncoderAttentionHeads { get; set; } = 16;

    [JsonPropertyName("encoder_ffn_dim")]
    public int EncoderFfnDim { get; set; } = 4096;

    [JsonPropertyName("dropout")]
    public float Dropout { get; set; } = 0.0f;

    [JsonPropertyName("attention_dropout")]
    public float AttentionDropout { get; set; } = 0.0f;

    [JsonPropertyName("activation_dropout")]
    public float ActivationDropout { get; set; } = 0.0f;

    [JsonPropertyName("activation_function")]
    public string ActivationFunction { get; set; } = "gelu";

    [JsonPropertyName("max_source_positions")]
    public int MaxSourcePositions { get; set; } = 1500;

    [JsonPropertyName("scale_embedding")]
    public bool ScaleEmbedding { get; set; } = false;
    
    [JsonPropertyName("encoder_layerdrop")]
    public float EncoderLayerDrop { get; set; } = 0.0f;

    [JsonPropertyName("n_window")]
    public int NWindow { get; set; } = 0;

    [JsonPropertyName("output_dim")]
    public int OutputDim { get; set; } = 0;

    public bool UseOmniAudioChunking { get; set; } = false;

    public bool UseSinusoidalPositionEmbedding { get; set; } = false;

    public bool ProjectOutput { get; set; } = false;
}

public class Qwen2AudioConfig
{
    [JsonPropertyName("audio_config")]
    public Qwen2AudioEncoderConfig AudioConfig { get; set; } = new();

    [JsonPropertyName("text_config")]
    public Qwen2Config TextConfig { get; set; } = new();

    [JsonPropertyName("audio_token_index")]
    public int AudioTokenIndex { get; set; } = 151646;

    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; } = 151936;
    
    [JsonPropertyName("ignore_index")]
    public int IgnoreIndex { get; set; } = -100;
}
