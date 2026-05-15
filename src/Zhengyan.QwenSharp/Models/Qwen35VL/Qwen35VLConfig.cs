using System.Text.Json.Serialization;
using Zhengyan.QwenSharp.Models.Qwen3VL;
using Zhengyan.QwenSharp.Models.Qwen35;

namespace Zhengyan.QwenSharp.Models.Qwen35VL;

public class Qwen35VLConfig
{
    [JsonPropertyName("vision_config")]
    public Qwen3VLVisionConfig VisionConfig { get; set; } = new();

    [JsonPropertyName("text_config")]
    public Qwen35Config TextConfig { get; set; } = new();

    [JsonPropertyName("image_token_id")]
    public int ImageTokenId { get; set; }

    [JsonPropertyName("video_token_id")]
    public int VideoTokenId { get; set; }

    [JsonPropertyName("vision_start_token_id")]
    public int VisionStartTokenId { get; set; }

    [JsonPropertyName("vision_end_token_id")]
    public int VisionEndTokenId { get; set; }
}

