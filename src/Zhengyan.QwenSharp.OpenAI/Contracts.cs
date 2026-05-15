using System.Text.Json;
using System.Text.Json.Serialization;

namespace Zhengyan.QwenSharp.OpenAI;

[JsonConverter(typeof(OpenAIContentPartJsonConverter))]
public sealed record OpenAIContentPart
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "text";

    [JsonPropertyName("text")]
    public string? Text { get; init; }

    [JsonPropertyName("audio")]
    public OpenAIInputAudio? Audio { get; init; }

    [JsonPropertyName("image_url")]
    public OpenAIImageUrl? ImageUrl { get; init; }

    [JsonPropertyName("video")]
    public OpenAIVideoReference? Video { get; init; }

    [JsonPropertyName("file")]
    public OpenAIFileReference? File { get; init; }
}

public sealed record OpenAIInputAudio
{
    [JsonPropertyName("data")]
    public string DataBase64 { get; init; } = "";

    [JsonPropertyName("format")]
    public string Format { get; init; } = "wav";
}

public sealed record OpenAIImageUrl
{
    [JsonPropertyName("url")]
    public string Url { get; init; } = "";

    [JsonPropertyName("file_id")]
    public string? FileId { get; init; }
}

public sealed record OpenAIVideoReference
{
    [JsonPropertyName("url")]
    public string? Url { get; init; }

    [JsonPropertyName("file_id")]
    public string? FileId { get; init; }

    [JsonPropertyName("data")]
    public string? DataBase64 { get; init; }

    [JsonPropertyName("format")]
    public string? Format { get; init; }

    [JsonPropertyName("fps")]
    public double? Fps { get; init; }

    [JsonPropertyName("nframes")]
    public int? NFrames { get; init; }

    [JsonPropertyName("video_start")]
    public double? VideoStart { get; init; }

    [JsonPropertyName("video_end")]
    public double? VideoEnd { get; init; }
}

public sealed record OpenAIFileReference
{
    [JsonPropertyName("file_id")]
    public string FileId { get; init; } = "";
}

[JsonConverter(typeof(OpenAIMessageJsonConverter))]
public sealed record OpenAIMessage
{
    [JsonPropertyName("role")]
    public string Role { get; init; } = "user";

    [JsonPropertyName("name")]
    public string? Name { get; init; }

    [JsonPropertyName("content")]
    public string? Content { get; init; }

    [JsonPropertyName("parts")]
    public IReadOnlyList<OpenAIContentPart>? Parts { get; init; }

    [JsonPropertyName("tool_call_id")]
    public string? ToolCallId { get; init; }

    [JsonPropertyName("tool_calls")]
    public IReadOnlyList<OpenAIToolCall>? ToolCalls { get; init; }

    [JsonPropertyName("function_call")]
    public OpenAIFunctionCall? FunctionCall { get; init; }
}

public sealed record OpenAIChatCompletionRequest
{
    [JsonPropertyName("model")]
    public string Model { get; init; } = "";

    [JsonPropertyName("messages")]
    public IReadOnlyList<OpenAIMessage> Messages { get; init; } = Array.Empty<OpenAIMessage>();

    [JsonPropertyName("temperature")]
    public float? Temperature { get; init; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; init; }

    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; init; }

    [JsonPropertyName("tools")]
    public IReadOnlyList<OpenAITool>? Tools { get; init; }

    [JsonPropertyName("tool_choice")]
    public JsonElement? ToolChoice { get; init; }

    [JsonPropertyName("parallel_tool_calls")]
    public bool? ParallelToolCalls { get; init; }

    [JsonPropertyName("response_format")]
    public JsonElement? ResponseFormat { get; init; }

    [JsonPropertyName("use_audio_in_video")]
    public bool? UseAudioInVideo { get; init; }
}

public sealed record OpenAIChatCompletionChoice
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("message")]
    public OpenAIMessage Message { get; init; } = new();

    [JsonPropertyName("finish_reason")]
    public string FinishReason { get; init; } = "stop";
}

public sealed record OpenAIChatCompletionResponse
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = $"chatcmpl_{Guid.NewGuid():N}";

    [JsonPropertyName("object")]
    public string Object { get; init; } = "chat.completion";

    [JsonPropertyName("created")]
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    [JsonPropertyName("model")]
    public string Model { get; init; } = "";

    [JsonPropertyName("choices")]
    public IReadOnlyList<OpenAIChatCompletionChoice> Choices { get; init; } = Array.Empty<OpenAIChatCompletionChoice>();

    [JsonPropertyName("warnings")]
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();
}

[JsonConverter(typeof(OpenAIResponseRequestJsonConverter))]
public sealed record OpenAIResponseRequest
{
    [JsonPropertyName("model")]
    public string Model { get; init; } = "";

    [JsonPropertyName("input")]
    public IReadOnlyList<OpenAIMessage> Input { get; init; } = Array.Empty<OpenAIMessage>();

    [JsonPropertyName("max_output_tokens")]
    public int? MaxOutputTokens { get; init; }

    [JsonPropertyName("voice")]
    public string? Voice { get; init; }

    [JsonPropertyName("temperature")]
    public float? Temperature { get; init; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; init; }

    [JsonPropertyName("use_audio_in_video")]
    public bool? UseAudioInVideo { get; init; }

    [JsonPropertyName("include")]
    public IReadOnlyList<string>? Include { get; init; }

    [JsonPropertyName("modalities")]
    public IReadOnlyList<string>? Modalities { get; init; }

    [JsonPropertyName("tools")]
    public IReadOnlyList<OpenAITool>? Tools { get; init; }

    [JsonPropertyName("tool_choice")]
    public JsonElement? ToolChoice { get; init; }

    [JsonPropertyName("parallel_tool_calls")]
    public bool? ParallelToolCalls { get; init; }

    [JsonPropertyName("instructions")]
    public string? Instructions { get; init; }

    [JsonPropertyName("text")]
    public JsonElement? Text { get; init; }
}

public sealed record OpenAIResponseOutput
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = $"msg_{Guid.NewGuid():N}";

    [JsonPropertyName("type")]
    public string Type { get; init; } = "message";

    [JsonPropertyName("role")]
    public string Role { get; init; } = "assistant";

    [JsonPropertyName("content")]
    public string Content { get; init; } = "";

    [JsonPropertyName("audio_base64")]
    public string? AudioBase64 { get; init; }

    [JsonPropertyName("audio_format")]
    public string? AudioFormat { get; init; }

    [JsonPropertyName("transcript")]
    public string? Transcript { get; init; }

    [JsonPropertyName("name")]
    public string? Name { get; init; }

    [JsonPropertyName("arguments")]
    public string? Arguments { get; init; }

    [JsonPropertyName("call_id")]
    public string? CallId { get; init; }

    [JsonPropertyName("tool_call_id")]
    public string? ToolCallId { get; init; }
}

public sealed record OpenAIResponseResponse
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = $"resp_{Guid.NewGuid():N}";

    [JsonPropertyName("object")]
    public string Object { get; init; } = "response";

    [JsonPropertyName("created_at")]
    public long CreatedAt { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    [JsonPropertyName("model")]
    public string Model { get; init; } = "";

    [JsonPropertyName("output")]
    public IReadOnlyList<OpenAIResponseOutput> Output { get; init; } = Array.Empty<OpenAIResponseOutput>();

    [JsonPropertyName("warnings")]
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();
}

public sealed record OpenAIAudioSpeechRequest
{
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("input")]
    public string Input { get; init; } = "";

    [JsonPropertyName("voice")]
    public string? Voice { get; init; }

    [JsonPropertyName("response_format")]
    public string ResponseFormat { get; init; } = "wav";

    [JsonPropertyName("speed")]
    public float? Speed { get; init; }
}

public sealed record OpenAIRealtimeAudioFormat(
    [property: JsonPropertyName("type")] string Type,
    [property: JsonPropertyName("rate")] int? Rate = null);

public sealed record OpenAIRealtimeInputAudioTranscription
{
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("language")]
    public string? Language { get; init; }

    [JsonPropertyName("prompt")]
    public string? Prompt { get; init; }
}

public sealed record OpenAIRealtimeTurnDetection
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "server_vad";

    [JsonPropertyName("threshold")]
    public float? Threshold { get; init; }

    [JsonPropertyName("prefix_padding_ms")]
    public int? PrefixPaddingMilliseconds { get; init; }

    [JsonPropertyName("silence_duration_ms")]
    public int? SilenceDurationMilliseconds { get; init; }

    [JsonPropertyName("idle_timeout_ms")]
    public int? IdleTimeoutMilliseconds { get; init; }
}

public sealed record OpenAIRealtimeSessionAudioOptions
{
    [JsonPropertyName("input")]
    public OpenAIRealtimeSessionInputAudioOptions Input { get; init; } = new();

    [JsonPropertyName("output")]
    public OpenAIRealtimeSessionOutputAudioOptions Output { get; init; } = new();
}

public sealed record OpenAIRealtimeSessionInputAudioOptions
{
    [JsonPropertyName("format")]
    public OpenAIRealtimeAudioFormat Format { get; init; } = new("audio/pcm", 16_000);

    [JsonPropertyName("transcription")]
    public OpenAIRealtimeInputAudioTranscription? Transcription { get; init; }

    [JsonPropertyName("turn_detection")]
    public OpenAIRealtimeTurnDetection? TurnDetection { get; init; }
}

public sealed record OpenAIRealtimeSessionOutputAudioOptions
{
    [JsonPropertyName("format")]
    public OpenAIRealtimeAudioFormat Format { get; init; } = new("audio/pcm", 24_000);

    [JsonPropertyName("voice")]
    public string? Voice { get; init; }
}

public sealed record OpenAIRealtimeSessionOptions
{
    [JsonPropertyName("id")]
    public string? Id { get; init; }

    [JsonPropertyName("model")]
    public string Model { get; init; } = "";

    [JsonPropertyName("instructions")]
    public string? Instructions { get; init; }

    [JsonPropertyName("output_modalities")]
    public IReadOnlyList<string>? OutputModalities { get; init; }

    [JsonPropertyName("audio")]
    public OpenAIRealtimeSessionAudioOptions Audio { get; init; } = new();

    [JsonPropertyName("voice")]
    public string? Voice { get; init; }

    [JsonPropertyName("sample_rate")]
    public int SampleRate { get; init; } = 16_000;

    [JsonPropertyName("return_audio")]
    public bool ReturnAudio { get; init; } = true;

    [JsonPropertyName("temperature")]
    public float? Temperature { get; init; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; init; }

    [JsonPropertyName("max_output_tokens")]
    public int? MaxOutputTokens { get; init; }

    [JsonPropertyName("turn_detection")]
    public string? TurnDetection { get; init; }

    [JsonPropertyName("use_audio_in_video")]
    public bool UseAudioInVideo { get; init; } = true;

    [JsonPropertyName("auto_respond")]
    public bool AutoRespond { get; init; } = true;
}

public sealed record OpenAIRealtimeEvent
{
    [JsonPropertyName("event_id")]
    public string EventId { get; init; } = $"evt_{Guid.NewGuid():N}";

    [JsonPropertyName("type")]
    public string Type { get; init; } = "";

    [JsonPropertyName("session")]
    public OpenAIRealtimeSessionOptions? Session { get; init; }

    [JsonPropertyName("response")]
    public OpenAIRealtimeResponseInfo? Response { get; init; }

    [JsonPropertyName("item")]
    public OpenAIRealtimeConversationItem? Item { get; init; }

    [JsonPropertyName("part")]
    public OpenAIRealtimeContentPart? Part { get; init; }

    [JsonPropertyName("session_id")]
    public string? SessionId { get; init; }

    [JsonPropertyName("item_id")]
    public string? ItemId { get; init; }

    [JsonPropertyName("response_id")]
    public string? ResponseId { get; init; }

    [JsonPropertyName("previous_item_id")]
    public string? PreviousItemId { get; init; }

    [JsonPropertyName("audio")]
    public string? Audio { get; init; }

    [JsonPropertyName("text")]
    public string? Text { get; init; }

    [JsonPropertyName("delta")]
    public string? Delta { get; init; }

    [JsonPropertyName("audio_base64")]
    public string? AudioBase64 { get; init; }

    [JsonPropertyName("format")]
    public string? Format { get; init; }

    [JsonPropertyName("sample_rate")]
    public int? SampleRate { get; init; }

    [JsonPropertyName("voice")]
    public string? Voice { get; init; }

    [JsonPropertyName("return_audio")]
    public bool? ReturnAudio { get; init; }

    [JsonPropertyName("instructions")]
    public string? Instructions { get; init; }

    [JsonPropertyName("temperature")]
    public float? Temperature { get; init; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; init; }

    [JsonPropertyName("max_output_tokens")]
    public int? MaxOutputTokens { get; init; }

    [JsonPropertyName("use_audio_in_video")]
    public bool? UseAudioInVideo { get; init; }

    [JsonPropertyName("auto_respond")]
    public bool? AutoRespond { get; init; }

    [JsonPropertyName("role")]
    public string? Role { get; init; }

    [JsonPropertyName("parts")]
    public IReadOnlyList<OpenAIContentPart>? Parts { get; init; }

    [JsonPropertyName("status")]
    public string? Status { get; init; }

    [JsonPropertyName("transcript")]
    public string? Transcript { get; init; }

    [JsonPropertyName("output_index")]
    public int? OutputIndex { get; init; }

    [JsonPropertyName("content_index")]
    public int? ContentIndex { get; init; }

    [JsonPropertyName("done")]
    public bool? Done { get; init; }

    [JsonPropertyName("error")]
    public OpenAIRealtimeError? Error { get; init; }

    [JsonPropertyName("warning")]
    public string? Warning { get; init; }

    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonIgnore]
    internal string? AudioRawBase64 { get; init; }

    [JsonIgnore]
    internal OpenAIRealtimeSessionOptions? SessionRequest { get; init; }

    [JsonIgnore]
    internal OpenAIRealtimeConversationItem? ItemRequest { get; init; }

    [JsonIgnore]
    internal OpenAIRealtimeResponseRequest? ResponseRequest { get; init; }
}

public sealed record OpenAIRealtimeConversationItem
{
    [JsonPropertyName("id")]
    public string? Id { get; init; }

    [JsonPropertyName("type")]
    public string Type { get; init; } = "message";

    [JsonPropertyName("status")]
    public string? Status { get; init; } = "completed";

    [JsonPropertyName("role")]
    public string? Role { get; init; }

    [JsonPropertyName("content")]
    public IReadOnlyList<OpenAIRealtimeContentPart> Content { get; init; } = Array.Empty<OpenAIRealtimeContentPart>();
}

public sealed record OpenAIRealtimeContentPart
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "text";

    [JsonPropertyName("text")]
    public string? Text { get; init; }

    [JsonPropertyName("transcript")]
    public string? Transcript { get; init; }

    [JsonPropertyName("audio")]
    public string? Audio { get; init; }
}

public sealed record OpenAIRealtimeResponseAudioOptions
{
    [JsonPropertyName("format")]
    public OpenAIRealtimeAudioFormat? Format { get; init; }

    [JsonPropertyName("voice")]
    public string? Voice { get; init; }
}

public sealed record OpenAIRealtimeResponseRequest
{
    [JsonPropertyName("conversation")]
    public string? Conversation { get; init; } = "auto";

    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("instructions")]
    public string? Instructions { get; init; }

    [JsonPropertyName("output_modalities")]
    public IReadOnlyList<string>? OutputModalities { get; init; }

    [JsonPropertyName("audio")]
    public OpenAIRealtimeResponseAudioOptions? Audio { get; init; }

    [JsonPropertyName("max_output_tokens")]
    public int? MaxOutputTokens { get; init; }

    [JsonPropertyName("temperature")]
    public float? Temperature { get; init; }
}

public sealed record OpenAIRealtimeResponseInfo
{
    [JsonPropertyName("id")]
    public string? Id { get; init; }

    [JsonPropertyName("status")]
    public string? Status { get; init; }

    [JsonPropertyName("output")]
    public IReadOnlyList<OpenAIRealtimeConversationItem> Output { get; init; } = Array.Empty<OpenAIRealtimeConversationItem>();

    [JsonPropertyName("conversation")]
    public string? Conversation { get; init; }

    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("instructions")]
    public string? Instructions { get; init; }

    [JsonPropertyName("output_modalities")]
    public IReadOnlyList<string>? OutputModalities { get; init; }

    [JsonPropertyName("audio")]
    public OpenAIRealtimeResponseAudioOptions? Audio { get; init; }

    [JsonPropertyName("max_output_tokens")]
    public int? MaxOutputTokens { get; init; }

    [JsonPropertyName("temperature")]
    public float? Temperature { get; init; }
}

public sealed record OpenAIRealtimeError
{
    [JsonPropertyName("type")]
    public string? Type { get; init; }

    [JsonPropertyName("code")]
    public string? Code { get; init; }

    [JsonPropertyName("message")]
    public string Message { get; init; } = "";

    [JsonPropertyName("event_id")]
    public string? EventId { get; init; }
}

public interface IOpenAIChatCompletionsService
{
    Task<OpenAIChatCompletionResponse> CreateCompletionAsync(OpenAIChatCompletionRequest request, CancellationToken cancellationToken = default);
}

public interface IOpenAIResponsesService
{
    Task<OpenAIResponseResponse> CreateResponseAsync(OpenAIResponseRequest request, CancellationToken cancellationToken = default);
}

public interface IOpenAIAudioSpeechService
{
    Task<(byte[] AudioBytes, string ContentType)> CreateSpeechAsync(OpenAIAudioSpeechRequest request, CancellationToken cancellationToken = default);
}

public interface IOpenAIRealtimeSession : IAsyncDisposable
{
    IAsyncEnumerable<OpenAIRealtimeEvent> Events { get; }

    ValueTask SendAsync(OpenAIRealtimeEvent evt, CancellationToken cancellationToken = default);
}

public interface IOpenAIRealtimeSessionFactory
{
    ValueTask<IOpenAIRealtimeSession> CreateAsync(OpenAIRealtimeSessionOptions options, CancellationToken cancellationToken = default);
}

public sealed record OpenAITool
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "function";

    [JsonPropertyName("function")]
    public OpenAIFunctionDefinition? Function { get; init; }
}

public sealed record OpenAIFunctionDefinition
{
    [JsonPropertyName("name")]
    public string Name { get; init; } = "";

    [JsonPropertyName("description")]
    public string? Description { get; init; }

    [JsonPropertyName("parameters")]
    public JsonElement? Parameters { get; init; }

    [JsonPropertyName("strict")]
    public bool? Strict { get; init; }
}

public sealed record OpenAIToolCall
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = $"call_{Guid.NewGuid():N}";

    [JsonPropertyName("type")]
    public string Type { get; init; } = "function";

    [JsonPropertyName("function")]
    public OpenAIFunctionCall Function { get; init; } = new();
}

public sealed record OpenAIFunctionCall
{
    [JsonPropertyName("name")]
    public string Name { get; init; } = "";

    [JsonPropertyName("arguments")]
    public string Arguments { get; init; } = "{}";
}

internal sealed class OpenAIContentPartJsonConverter : JsonConverter<OpenAIContentPart>
{
    public override OpenAIContentPart Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        using var document = JsonDocument.ParseValue(ref reader);
        var root = document.RootElement;
        var type = ReadString(root, "type") ?? "text";
        OpenAIInputAudio? audio = null;
        OpenAIImageUrl? imageUrl = null;
        OpenAIVideoReference? video = null;
        OpenAIFileReference? file = null;

        var text = ReadString(root, "text");
        if (root.TryGetProperty("audio", out var audioElement) && audioElement.ValueKind == JsonValueKind.Object)
        {
            audio = audioElement.Deserialize<OpenAIInputAudio>(options);
        }

        if (root.TryGetProperty("input_audio", out var inputAudioElement) && inputAudioElement.ValueKind == JsonValueKind.Object)
        {
            audio = inputAudioElement.Deserialize<OpenAIInputAudio>(options);
        }

        if (root.TryGetProperty("image_url", out var imageElement))
        {
            imageUrl = ReadImageUrl(imageElement, options);
        }

        if (root.TryGetProperty("video", out var videoElement) && videoElement.ValueKind == JsonValueKind.Object)
        {
            video = videoElement.Deserialize<OpenAIVideoReference>(options);
        }

        if (root.TryGetProperty("file", out var fileElement) && fileElement.ValueKind == JsonValueKind.Object)
        {
            file = fileElement.Deserialize<OpenAIFileReference>(options);
        }

        if (root.TryGetProperty("file_id", out var fileIdElement) && fileIdElement.ValueKind == JsonValueKind.String)
        {
            var fileId = fileIdElement.GetString() ?? string.Empty;
            if (string.Equals(type, "input_image", StringComparison.OrdinalIgnoreCase))
            {
                imageUrl ??= new OpenAIImageUrl { FileId = fileId };
            }
            else
            {
                file ??= new OpenAIFileReference { FileId = fileId };
            }
        }

        var normalizedType = type switch
        {
            "input_text" => "text",
            "output_text" => "text",
            "input_image" => "image_url",
            _ => type,
        };

        return new OpenAIContentPart
        {
            Type = normalizedType,
            Text = text,
            Audio = audio,
            ImageUrl = imageUrl,
            Video = video,
            File = file,
        };
    }

    public override void Write(Utf8JsonWriter writer, OpenAIContentPart value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WriteString("type", value.Type);
        if (value.Text is not null)
        {
            writer.WriteString("text", value.Text);
        }

        if (value.Audio is not null)
        {
            writer.WritePropertyName("audio");
            JsonSerializer.Serialize(writer, value.Audio, options);
        }

        if (value.ImageUrl is not null)
        {
            writer.WritePropertyName("image_url");
            JsonSerializer.Serialize(writer, value.ImageUrl, options);
        }

        if (value.Video is not null)
        {
            writer.WritePropertyName("video");
            JsonSerializer.Serialize(writer, value.Video, options);
        }

        if (value.File is not null)
        {
            writer.WritePropertyName("file");
            JsonSerializer.Serialize(writer, value.File, options);
        }

        writer.WriteEndObject();
    }

    private static OpenAIImageUrl? ReadImageUrl(JsonElement element, JsonSerializerOptions options)
        => element.ValueKind switch
        {
            JsonValueKind.String => new OpenAIImageUrl { Url = element.GetString() ?? string.Empty },
            JsonValueKind.Object => element.Deserialize<OpenAIImageUrl>(options),
            _ => null,
        };

    private static string? ReadString(JsonElement element, string propertyName)
        => element.TryGetProperty(propertyName, out var property) && property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;
}

internal sealed class OpenAIMessageJsonConverter : JsonConverter<OpenAIMessage>
{
    public override OpenAIMessage Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        using var document = JsonDocument.ParseValue(ref reader);
        var root = document.RootElement;
        var role = ReadString(root, "role") ?? "user";
        var name = ReadString(root, "name");
        var toolCallId = ReadString(root, "tool_call_id");
        string? content = null;
        IReadOnlyList<OpenAIContentPart>? parts = null;

        if (root.TryGetProperty("content", out var contentElement))
        {
            if (contentElement.ValueKind == JsonValueKind.String)
            {
                content = contentElement.GetString();
            }
            else if (contentElement.ValueKind == JsonValueKind.Array)
            {
                parts = contentElement.Deserialize<IReadOnlyList<OpenAIContentPart>>(options);
                content = FlattenParts(parts);
            }
            else if (contentElement.ValueKind == JsonValueKind.Null)
            {
                content = null;
            }
        }

        if (root.TryGetProperty("parts", out var partsElement) && partsElement.ValueKind == JsonValueKind.Array)
        {
            parts = partsElement.Deserialize<IReadOnlyList<OpenAIContentPart>>(options);
            content ??= FlattenParts(parts);
        }

        var toolCalls = root.TryGetProperty("tool_calls", out var toolCallsElement) && toolCallsElement.ValueKind == JsonValueKind.Array
            ? toolCallsElement.Deserialize<IReadOnlyList<OpenAIToolCall>>(options)
            : null;
        var functionCall = root.TryGetProperty("function_call", out var functionCallElement) && functionCallElement.ValueKind == JsonValueKind.Object
            ? functionCallElement.Deserialize<OpenAIFunctionCall>(options)
            : null;

        return new OpenAIMessage
        {
            Role = role,
            Name = name,
            Content = content,
            Parts = parts,
            ToolCallId = toolCallId,
            ToolCalls = toolCalls,
            FunctionCall = functionCall,
        };
    }

    public override void Write(Utf8JsonWriter writer, OpenAIMessage value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WriteString("role", value.Role);
        if (value.Name is not null)
        {
            writer.WriteString("name", value.Name);
        }

        if (value.Parts is not null && value.Parts.Count > 0)
        {
            writer.WritePropertyName("content");
            JsonSerializer.Serialize(writer, value.Parts, options);
        }
        else if (value.Content is not null)
        {
            writer.WriteString("content", value.Content);
        }
        else
        {
            writer.WriteNull("content");
        }

        if (value.ToolCallId is not null)
        {
            writer.WriteString("tool_call_id", value.ToolCallId);
        }

        if (value.ToolCalls is not null)
        {
            writer.WritePropertyName("tool_calls");
            JsonSerializer.Serialize(writer, value.ToolCalls, options);
        }

        if (value.FunctionCall is not null)
        {
            writer.WritePropertyName("function_call");
            JsonSerializer.Serialize(writer, value.FunctionCall, options);
        }

        writer.WriteEndObject();
    }

    private static string? ReadString(JsonElement element, string propertyName)
        => element.TryGetProperty(propertyName, out var property) && property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;

    private static string? FlattenParts(IReadOnlyList<OpenAIContentPart>? parts)
    {
        if (parts is null)
        {
            return null;
        }

        var text = string.Concat(parts
            .Where(part => string.Equals(part.Type, "text", StringComparison.OrdinalIgnoreCase))
            .Select(part => part.Text));
        return string.IsNullOrEmpty(text) ? null : text;
    }
}

internal sealed class OpenAIResponseRequestJsonConverter : JsonConverter<OpenAIResponseRequest>
{
    public override OpenAIResponseRequest Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        using var document = JsonDocument.ParseValue(ref reader);
        var root = document.RootElement;
        return new OpenAIResponseRequest
        {
            Model = ReadString(root, "model") ?? string.Empty,
            Input = ReadInput(root, options),
            MaxOutputTokens = ReadInt(root, "max_output_tokens"),
            Voice = ReadString(root, "voice"),
            Temperature = ReadFloat(root, "temperature"),
            TopP = ReadFloat(root, "top_p"),
            UseAudioInVideo = ReadBool(root, "use_audio_in_video"),
            Include = ReadStringArray(root, "include"),
            Modalities = ReadStringArray(root, "modalities"),
            Tools = ReadTools(root, options),
            ToolChoice = CloneElement(root, "tool_choice"),
            ParallelToolCalls = ReadBool(root, "parallel_tool_calls"),
            Instructions = ReadString(root, "instructions"),
            Text = CloneElement(root, "text"),
        };
    }

    public override void Write(Utf8JsonWriter writer, OpenAIResponseRequest value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WriteString("model", value.Model);
        writer.WritePropertyName("input");
        JsonSerializer.Serialize(writer, value.Input, options);
        WriteNullableNumber(writer, "max_output_tokens", value.MaxOutputTokens);
        WriteNullableString(writer, "voice", value.Voice);
        WriteNullableNumber(writer, "temperature", value.Temperature);
        WriteNullableNumber(writer, "top_p", value.TopP);
        WriteNullableBool(writer, "use_audio_in_video", value.UseAudioInVideo);
        WriteNullableArray(writer, "include", value.Include, options);
        WriteNullableArray(writer, "modalities", value.Modalities, options);
        WriteNullableArray(writer, "tools", value.Tools, options);
        WriteNullableElement(writer, "tool_choice", value.ToolChoice);
        WriteNullableBool(writer, "parallel_tool_calls", value.ParallelToolCalls);
        WriteNullableString(writer, "instructions", value.Instructions);
        WriteNullableElement(writer, "text", value.Text);
        writer.WriteEndObject();
    }

    private static IReadOnlyList<OpenAIMessage> ReadInput(JsonElement root, JsonSerializerOptions options)
    {
        if (!root.TryGetProperty("input", out var input))
        {
            return Array.Empty<OpenAIMessage>();
        }

        if (input.ValueKind == JsonValueKind.String)
        {
            return [new OpenAIMessage { Role = "user", Content = input.GetString() ?? string.Empty }];
        }

        if (input.ValueKind == JsonValueKind.Object)
        {
            return [ReadInputObject(input, options)];
        }

        if (input.ValueKind != JsonValueKind.Array)
        {
            return Array.Empty<OpenAIMessage>();
        }

        var messages = new List<OpenAIMessage>();
        foreach (var item in input.EnumerateArray())
        {
            if (item.ValueKind == JsonValueKind.String)
            {
                messages.Add(new OpenAIMessage { Role = "user", Content = item.GetString() ?? string.Empty });
            }
            else if (item.ValueKind == JsonValueKind.Object)
            {
                messages.Add(ReadInputObject(item, options));
            }
        }

        return messages;
    }

    private static OpenAIMessage ReadInputObject(JsonElement item, JsonSerializerOptions options)
    {
        if (!item.TryGetProperty("role", out _))
        {
            using var document = JsonDocument.Parse(item.GetRawText());
            var patched = new Dictionary<string, JsonElement>
            {
                ["role"] = JsonDocument.Parse("\"user\"").RootElement.Clone(),
            };

            foreach (var property in document.RootElement.EnumerateObject())
            {
                patched[property.Name] = property.Value.Clone();
            }

            var json = JsonSerializer.Serialize(patched, options);
            return JsonSerializer.Deserialize<OpenAIMessage>(json, options) ?? new OpenAIMessage();
        }

        return item.Deserialize<OpenAIMessage>(options) ?? new OpenAIMessage();
    }

    private static IReadOnlyList<OpenAITool>? ReadTools(JsonElement root, JsonSerializerOptions options)
        => root.TryGetProperty("tools", out var tools) && tools.ValueKind == JsonValueKind.Array
            ? tools.Deserialize<IReadOnlyList<OpenAITool>>(options)
            : null;

    private static IReadOnlyList<string>? ReadStringArray(JsonElement root, string propertyName)
        => root.TryGetProperty(propertyName, out var property) && property.ValueKind == JsonValueKind.Array
            ? property.EnumerateArray()
                .Where(item => item.ValueKind == JsonValueKind.String)
                .Select(item => item.GetString()!)
                .ToArray()
            : null;

    private static JsonElement? CloneElement(JsonElement root, string propertyName)
        => root.TryGetProperty(propertyName, out var property) && property.ValueKind != JsonValueKind.Null
            ? property.Clone()
            : null;

    private static string? ReadString(JsonElement element, string propertyName)
        => element.TryGetProperty(propertyName, out var property) && property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;

    private static int? ReadInt(JsonElement element, string propertyName)
        => element.TryGetProperty(propertyName, out var property) && property.TryGetInt32(out var value)
            ? value
            : null;

    private static float? ReadFloat(JsonElement element, string propertyName)
        => element.TryGetProperty(propertyName, out var property) && property.TryGetSingle(out var value)
            ? value
            : null;

    private static bool? ReadBool(JsonElement element, string propertyName)
        => element.TryGetProperty(propertyName, out var property) && property.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? property.GetBoolean()
            : null;

    private static void WriteNullableString(Utf8JsonWriter writer, string propertyName, string? value)
    {
        if (value is not null)
        {
            writer.WriteString(propertyName, value);
        }
    }

    private static void WriteNullableNumber(Utf8JsonWriter writer, string propertyName, int? value)
    {
        if (value is not null)
        {
            writer.WriteNumber(propertyName, value.Value);
        }
    }

    private static void WriteNullableNumber(Utf8JsonWriter writer, string propertyName, float? value)
    {
        if (value is not null)
        {
            writer.WriteNumber(propertyName, value.Value);
        }
    }

    private static void WriteNullableBool(Utf8JsonWriter writer, string propertyName, bool? value)
    {
        if (value is not null)
        {
            writer.WriteBoolean(propertyName, value.Value);
        }
    }

    private static void WriteNullableArray<T>(Utf8JsonWriter writer, string propertyName, IReadOnlyList<T>? value, JsonSerializerOptions options)
    {
        if (value is not null)
        {
            writer.WritePropertyName(propertyName);
            JsonSerializer.Serialize(writer, value, options);
        }
    }

    private static void WriteNullableElement(Utf8JsonWriter writer, string propertyName, JsonElement? value)
    {
        if (value is not null)
        {
            writer.WritePropertyName(propertyName);
            value.Value.WriteTo(writer);
        }
    }
}
