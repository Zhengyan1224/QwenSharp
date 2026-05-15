using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using Zhengyan.QwenSharp.Core;

namespace Zhengyan.QwenSharp.Models.Qwen25Omni;

/// <summary>
/// Configuration for the Qwen2.5-Omni Audio Encoder sub-module.
/// </summary>
public class Qwen25OmniAudioEncoderConfig : ModelConfig
{
    [JsonPropertyName("num_mel_bins")]
    public int NumMelBins { get; set; } = 128;

    [JsonPropertyName("encoder_layers")]
    public int EncoderLayers { get; set; } = 32;

    [JsonPropertyName("encoder_attention_heads")]
    public int EncoderAttentionHeads { get; set; } = 20;

    [JsonPropertyName("encoder_ffn_dim")]
    public int EncoderFfnDim { get; set; } = 5120;

    [JsonPropertyName("d_model")]
    public int DModel { get; set; } = 1280;

    [JsonPropertyName("dropout")]
    public double Dropout { get; set; } = 0.0;

    [JsonPropertyName("attention_dropout")]
    public double AttentionDropout { get; set; } = 0.0;

    [JsonPropertyName("activation_function")]
    public string ActivationFunction { get; set; } = "gelu";

    [JsonPropertyName("activation_dropout")]
    public double ActivationDropout { get; set; } = 0.0;

    [JsonPropertyName("scale_embedding")]
    public bool ScaleEmbedding { get; set; } = false;

    [JsonPropertyName("max_source_positions")]
    public int MaxSourcePositions { get; set; } = 1500;

    [JsonPropertyName("n_window")]
    public int NWindow { get; set; } = 100;

    [JsonPropertyName("output_dim")]
    public int OutputDim { get; set; } = 3584;

    public Qwen25OmniAudioEncoderConfig()
    {
        ModelType = "qwen2_5_omni_audio_encoder";
    }
}

/// <summary>
/// Configuration for the Qwen2.5-Omni Vision Encoder sub-module.
/// </summary>
public class Qwen25OmniVisionEncoderConfig : ModelConfig
{
    [JsonPropertyName("depth")]
    public int Depth { get; set; } = 32;

    [JsonPropertyName("hidden_size")]
    public new int HiddenSize { get; set; } = 3584;

    [JsonPropertyName("hidden_act")]
    public string HiddenAct { get; set; } = "silu";

    [JsonPropertyName("intermediate_size")]
    public new int IntermediateSize { get; set; } = 3420;

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

    [JsonPropertyName("window_size")]
    public int WindowSize { get; set; } = 112;

    [JsonPropertyName("out_hidden_size")]
    public int OutHiddenSize { get; set; } = 3584;

    [JsonPropertyName("fullatt_block_indexes")]
    public int[] FullattBlockIndexes { get; set; } = { 7, 15, 23, 31 };

    public Qwen25OmniVisionEncoderConfig()
    {
        ModelType = "qwen2_5_omni_vision_encoder";
    }
}

/// <summary>
/// Text (LLM decoder) configuration for the Qwen2.5-Omni Thinker.
/// </summary>
public class Qwen25OmniTextConfig : ModelConfig
{
    [JsonPropertyName("vocab_size")]
    public new int VocabSize { get; set; } = 152064;

    [JsonPropertyName("hidden_size")]
    public new int HiddenSize { get; set; } = 3584;

    [JsonPropertyName("intermediate_size")]
    public new int IntermediateSize { get; set; } = 18944;

    [JsonPropertyName("num_hidden_layers")]
    public new int NumHiddenLayers { get; set; } = 28;

    [JsonPropertyName("num_attention_heads")]
    public new int NumAttentionHeads { get; set; } = 28;

    [JsonPropertyName("num_key_value_heads")]
    public new int? NumKeyValueHeads { get; set; } = 4;

    [JsonPropertyName("hidden_act")]
    public string HiddenAct { get; set; } = "silu";

    [JsonPropertyName("max_position_embeddings")]
    public new int MaxPositionEmbeddings { get; set; } = 32768;

    [JsonPropertyName("rms_norm_eps")]
    public new float RmsNormEps { get; set; } = 1e-6f;

    [JsonPropertyName("use_sliding_window")]
    public new bool UseSlidingWindow { get; set; } = false;

    [JsonPropertyName("sliding_window")]
    public new int? SlidingWindow { get; set; } = 32768;

    [JsonPropertyName("max_window_layers")]
    public new int MaxWindowLayers { get; set; } = 28;

    [JsonPropertyName("tie_word_embeddings")]
    public new bool TieWordEmbeddings { get; set; } = true;

    public int[] MRopeSection
    {
        get
        {
            if (RopeScaling.HasValue &&
                RopeScaling.Value.ValueKind == JsonValueKind.Object &&
                RopeScaling.Value.TryGetProperty("mrope_section", out var sectionElement) &&
                sectionElement.ValueKind == JsonValueKind.Array)
            {
                return sectionElement.EnumerateArray().Select(section => section.GetInt32()).ToArray();
            }

            return [16, 24, 24];
        }
    }

    public Qwen25OmniTextConfig()
    {
        ModelType = "qwen2_5_omni_text";
    }
}

/// <summary>
/// Thinker configuration merging audio, vision, and text sub-module configurations.
/// </summary>
public class Qwen25OmniThinkerConfig : ModelConfig
{
    [JsonPropertyName("audio_token_index")]
    public int AudioTokenIndex { get; set; } = 151646;

    [JsonPropertyName("image_token_index")]
    public int ImageTokenIndex { get; set; } = 151655;

    [JsonPropertyName("video_token_index")]
    public int VideoTokenIndex { get; set; } = 151656;

    [JsonPropertyName("vision_start_token_id")]
    public int VisionStartTokenId { get; set; } = 151652;

    [JsonPropertyName("vision_end_token_id")]
    public int VisionEndTokenId { get; set; } = 151653;

    [JsonPropertyName("user_token_id")]
    public int UserTokenId { get; set; } = 872;

    [JsonPropertyName("position_id_per_seconds")]
    public int PositionIdPerSeconds { get; set; } = 25;

    [JsonPropertyName("seconds_per_chunk")]
    public int SecondsPerChunk { get; set; } = 2;

    [JsonPropertyName("audio_start_token_id")]
    public int AudioStartTokenId { get; set; } = 151647;

    [JsonPropertyName("audio_end_token_id")]
    public int AudioEndTokenId { get; set; } = 151648;

    [JsonPropertyName("tie_word_embeddings")]
    public new bool TieWordEmbeddings { get; set; } = false;

    [JsonPropertyName("audio_config")]
    public Qwen25OmniAudioEncoderConfig AudioConfig { get; set; } = new();

    [JsonPropertyName("vision_config")]
    public Qwen25OmniVisionEncoderConfig VisionConfig { get; set; } = new();

    [JsonPropertyName("text_config")]
    public Qwen25OmniTextConfig TextConfig { get; set; } = new();

    public Qwen25OmniThinkerConfig()
    {
        ModelType = "qwen2_5_omni_thinker";
    }
}

/// <summary>
/// Full Qwen2.5-Omni unified configuration.
/// Wraps the Thinker configuration used for text comprehension and generation.
/// </summary>
/// <summary>
/// Configuration for the Qwen2.5-Omni Talker sub-module (audio generation).
/// </summary>
public class Qwen25OmniTalkerConfig : ModelConfig
{
    [JsonPropertyName("hidden_size")]
    public new int HiddenSize { get; set; } = 3584;

    [JsonPropertyName("num_hidden_layers")]
    public new int NumHiddenLayers { get; set; } = 28;

    public Qwen25OmniTalkerConfig()
    {
        ModelType = "qwen2_5_omni_talker";
    }
}

/// <summary>
/// Configuration for the Qwen2.5-Omni Token2Wav sub-module (diffusion vocoder).
/// </summary>
public class Qwen25OmniToken2WavConfig : ModelConfig
{
    [JsonPropertyName("hidden_size")]
    public new int HiddenSize { get; set; } = 1024;

    [JsonPropertyName("num_hidden_layers")]
    public new int NumHiddenLayers { get; set; } = 12;

    [JsonPropertyName("vocoder_type")]
    public string VocoderType { get; set; } = "dit"; // "dit" or "bigvgan"

    public Qwen25OmniToken2WavConfig()
    {
        ModelType = "qwen2_5_omni_token2wav";
    }
}

public class Qwen25OmniConfig : ModelConfig
{
    [JsonPropertyName("thinker_config")]
    public Qwen25OmniThinkerConfig ThinkerConfig { get; set; } = new();

    public Qwen25OmniConfig()
    {
        ModelType = "qwen2_5_omni";
    }
}
