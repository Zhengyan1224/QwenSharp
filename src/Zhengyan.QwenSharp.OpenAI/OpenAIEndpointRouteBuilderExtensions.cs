using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;

namespace Zhengyan.QwenSharp.OpenAI;

public sealed class OpenAIEndpointOptions
{
    public string ChatCompletionsPath { get; init; } = "/v1/chat/completions";

    public string ResponsesPath { get; init; } = "/v1/responses";

    public string AudioSpeechPath { get; init; } = "/v1/audio/speech";

    public string RealtimePath { get; init; } = "/v1/realtime";

    public bool EnableRealtime { get; init; } = true;

    public int RealtimeReceiveBufferBytes { get; init; } = 64 * 1024;

    public int RealtimeMaxTextMessageBytes { get; init; } = 256 * 1024 * 1024;

    public JsonSerializerOptions JsonSerializerOptions { get; init; } = new()
    {
        PropertyNamingPolicy = null,
        WriteIndented = false,
    };

    public OpenAIRealtimeSessionOptions DefaultRealtimeSession { get; init; } = new()
    {
        Model = "qwen",
        SampleRate = 16_000,
        ReturnAudio = true,
        AutoRespond = true,
        UseAudioInVideo = true,
    };
}

public static class OpenAIEndpointRouteBuilderExtensions
{
    public static IEndpointRouteBuilder MapOpenAICompatibleApi(
        this IEndpointRouteBuilder endpoints,
        OpenAIEndpointOptions? options = null)
    {
        options ??= new OpenAIEndpointOptions();

        endpoints.MapPost(options.ChatCompletionsPath, async (
            OpenAIChatCompletionRequest request,
            IOpenAIChatCompletionsService service,
            CancellationToken cancellationToken)
            => Results.Json(await service.CreateCompletionAsync(request, cancellationToken).ConfigureAwait(false), options.JsonSerializerOptions));

        endpoints.MapPost(options.ResponsesPath, async (
            OpenAIResponseRequest request,
            IOpenAIResponsesService service,
            CancellationToken cancellationToken)
            => Results.Json(await service.CreateResponseAsync(request, cancellationToken).ConfigureAwait(false), options.JsonSerializerOptions));

        endpoints.MapPost(options.AudioSpeechPath, async (
            OpenAIAudioSpeechRequest request,
            IOpenAIAudioSpeechService service,
            CancellationToken cancellationToken) =>
        {
            var (audioBytes, contentType) = await service.CreateSpeechAsync(request, cancellationToken).ConfigureAwait(false);
            return Results.File(audioBytes, contentType);
        });

        if (options.EnableRealtime)
        {
            endpoints.Map(options.RealtimePath, context => HandleRealtimeWebSocketAsync(context, options));
        }

        return endpoints;
    }

    private static async Task HandleRealtimeWebSocketAsync(HttpContext context, OpenAIEndpointOptions options)
    {
        if (!context.WebSockets.IsWebSocketRequest)
        {
            context.Response.StatusCode = StatusCodes.Status400BadRequest;
            await context.Response.WriteAsync("WebSocket request required.", context.RequestAborted).ConfigureAwait(false);
            return;
        }

        using var socket = await context.WebSockets.AcceptWebSocketAsync().ConfigureAwait(false);
        var factory = context.RequestServices.GetRequiredService<IOpenAIRealtimeSessionFactory>();
        var session = await factory.CreateAsync(options.DefaultRealtimeSession, context.RequestAborted).ConfigureAwait(false);

        var outgoing = Task.Run(async () =>
        {
            await foreach (var evt in session.Events.WithCancellation(context.RequestAborted).ConfigureAwait(false))
            {
                await SendTextAsync(socket, evt, options.JsonSerializerOptions, context.RequestAborted).ConfigureAwait(false);
            }
        }, context.RequestAborted);

        try
        {
            while (socket.State == WebSocketState.Open && !context.RequestAborted.IsCancellationRequested)
            {
                var json = await ReceiveTextMessageAsync(socket, options, context.RequestAborted).ConfigureAwait(false);
                if (json is null)
                {
                    break;
                }

                if (json.Length == 0)
                {
                    continue;
                }

                OpenAIRealtimeEvent? evt;
                try
                {
                    evt = JsonSerializer.Deserialize<OpenAIRealtimeEvent>(json, options.JsonSerializerOptions);
                }
                catch (JsonException ex)
                {
                    await SendTextAsync(socket, new OpenAIRealtimeEvent
                    {
                        Type = "error",
                        Text = "Invalid realtime JSON message.",
                        Error = new OpenAIRealtimeError
                        {
                            Type = "invalid_request_error",
                            Message = ex.Message,
                        },
                        Done = true,
                    }, options.JsonSerializerOptions, context.RequestAborted).ConfigureAwait(false);
                    continue;
                }

                if (evt is not null)
                {
                    await session.SendAsync(evt, context.RequestAborted).ConfigureAwait(false);
                }
            }
        }
        catch (InvalidOperationException ex) when (socket.State == WebSocketState.Open && !context.RequestAborted.IsCancellationRequested)
        {
            await SendTextAsync(socket, new OpenAIRealtimeEvent
            {
                Type = "error",
                Text = ex.Message,
                Error = new OpenAIRealtimeError
                {
                    Type = "server_error",
                    Message = ex.Message,
                },
                Done = true,
            }, options.JsonSerializerOptions, context.RequestAborted).ConfigureAwait(false);
        }
        finally
        {
            await session.DisposeAsync().ConfigureAwait(false);

            if (socket.State == WebSocketState.Open)
            {
                await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "closed", context.RequestAborted).ConfigureAwait(false);
            }

            try
            {
                await outgoing.ConfigureAwait(false);
            }
            catch
            {
            }
        }
    }

    private static async Task<string?> ReceiveTextMessageAsync(
        WebSocket socket,
        OpenAIEndpointOptions options,
        CancellationToken cancellationToken)
    {
        var buffer = new byte[options.RealtimeReceiveBufferBytes];
        using var ms = new MemoryStream();

        while (true)
        {
            var result = await socket.ReceiveAsync(buffer, cancellationToken).ConfigureAwait(false);
            if (result.MessageType == WebSocketMessageType.Close)
            {
                return null;
            }

            if (result.MessageType != WebSocketMessageType.Text)
            {
                if (result.EndOfMessage)
                {
                    return string.Empty;
                }

                continue;
            }

            if (result.Count > 0)
            {
                ms.Write(buffer, 0, result.Count);
            }

            if (ms.Length > options.RealtimeMaxTextMessageBytes)
            {
                throw new InvalidOperationException(
                    $"Realtime text message exceeds the {options.RealtimeMaxTextMessageBytes / 1024 / 1024} MB limit. Use a smaller media clip or URL input.");
            }

            if (result.EndOfMessage)
            {
                return Encoding.UTF8.GetString(ms.ToArray());
            }
        }
    }

    private static async Task SendTextAsync(
        WebSocket socket,
        OpenAIRealtimeEvent evt,
        JsonSerializerOptions options,
        CancellationToken cancellationToken)
    {
        var json = JsonSerializer.Serialize(evt, options);
        var payload = Encoding.UTF8.GetBytes(json);
        await socket.SendAsync(payload, WebSocketMessageType.Text, true, cancellationToken).ConfigureAwait(false);
    }
}
