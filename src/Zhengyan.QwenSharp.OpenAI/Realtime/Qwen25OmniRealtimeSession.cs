using System.Text;
using System.Threading.Channels;
using Zhengyan.QwenSharp.Omni.Audio;
using Zhengyan.QwenSharp.OpenAI;

namespace Zhengyan.QwenSharp.OpenAI.Realtime;

public sealed class Qwen25OmniRealtimeSession : IOpenAIRealtimeSession
{
    private readonly Qwen25OmniOpenAIService _service;
    private readonly Channel<OpenAIRealtimeEvent> _events = Channel.CreateUnbounded<OpenAIRealtimeEvent>();
    private readonly List<OpenAIMessage> _conversation = [];
    private readonly List<float> _inputPcm = [];
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly object _stateLock = new();

    private bool _disposed;
    private string? _instructions;
    private int _sampleRate;
    private string? _voice;
    private bool _returnAudio;
    private bool _autoRespond;
    private float? _temperature;
    private float? _topP;
    private int? _maxOutputTokens;
    private bool _useAudioInVideo;
    private readonly string _model;
    private readonly string _sessionId = $"sess_{Guid.NewGuid():N}";
    private bool _speechStartedEmitted;
    private string? _lastItemId;
    private CancellationTokenSource? _generationCts;

    public Qwen25OmniRealtimeSession(Qwen25OmniOpenAIService service, OpenAIRealtimeSessionOptions options)
    {
        _service = service;
        _model = options.Model;
        _instructions = options.Instructions;
        _sampleRate = options.SampleRate > 0 ? options.SampleRate : 16_000;
        _voice = options.Voice;
        _returnAudio = options.ReturnAudio;
        _autoRespond = options.AutoRespond;
        _temperature = options.Temperature;
        _topP = options.TopP;
        _maxOutputTokens = options.MaxOutputTokens;
        _useAudioInVideo = options.UseAudioInVideo;
        _ = _events.Writer.TryWrite(new OpenAIRealtimeEvent
        {
            Type = "session.created",
            SessionId = _sessionId,
            SampleRate = _sampleRate,
            Voice = _voice,
            ReturnAudio = _returnAudio,
            Model = _model,
            Done = true,
        });
    }

    public IAsyncEnumerable<OpenAIRealtimeEvent> Events => _events.Reader.ReadAllAsync();

    public async ValueTask SendAsync(OpenAIRealtimeEvent evt, CancellationToken cancellationToken = default)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(Qwen25OmniRealtimeSession));
        }

        switch (evt.Type)
        {
            case "session.update":
                await HandleSessionUpdateAsync(evt, cancellationToken).ConfigureAwait(false);
                break;
            case "conversation.item.create":
                HandleConversationItemCreate(evt);
                break;
            case "input_audio_buffer.append":
                HandleAudioAppend(evt);
                break;
            case "input_audio_buffer.commit":
                await HandleAudioCommitAsync(cancellationToken).ConfigureAwait(false);
                break;
            case "response.create":
                await HandleResponseCreateAsync(evt, cancellationToken).ConfigureAwait(false);
                break;
            case "input_audio_buffer.clear":
                _inputPcm.Clear();
                _speechStartedEmitted = false;
                await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
                {
                    Type = "input_audio_buffer.cleared",
                    SessionId = _sessionId,
                    Done = true,
                }, cancellationToken).ConfigureAwait(false);
                break;
            case "response.cancel":
                _inputPcm.Clear();
                _speechStartedEmitted = false;
                CancelActiveGeneration();
                await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
                {
                    Type = "response.cancelled",
                    SessionId = _sessionId,
                    Done = true,
                }, cancellationToken).ConfigureAwait(false);
                break;
            default:
                await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
                {
                    Type = "error",
                    Text = $"Unsupported realtime event: {evt.Type}",
                    SessionId = _sessionId,
                    Done = true,
                }, cancellationToken).ConfigureAwait(false);
                break;
        }
    }

    private async Task HandleSessionUpdateAsync(OpenAIRealtimeEvent evt, CancellationToken cancellationToken)
    {
        if (evt.SampleRate is > 0)
        {
            _sampleRate = evt.SampleRate.Value;
        }

        if (!string.IsNullOrWhiteSpace(evt.Voice))
        {
            _voice = evt.Voice;
        }

        if (evt.Instructions is not null)
        {
            _instructions = evt.Instructions;
        }

        if (evt.ReturnAudio is not null)
        {
            _returnAudio = evt.ReturnAudio.Value;
        }

        if (evt.Temperature is not null)
        {
            _temperature = evt.Temperature.Value;
        }

        if (evt.TopP is not null)
        {
            _topP = evt.TopP.Value;
        }

        if (evt.MaxOutputTokens is not null)
        {
            _maxOutputTokens = evt.MaxOutputTokens.Value;
        }

        if (evt.UseAudioInVideo is not null)
        {
            _useAudioInVideo = evt.UseAudioInVideo.Value;
        }

        if (evt.AutoRespond is not null)
        {
            _autoRespond = evt.AutoRespond.Value;
        }

        await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
        {
            Type = "session.updated",
            SessionId = _sessionId,
            SampleRate = _sampleRate,
            Voice = _voice,
            ReturnAudio = _returnAudio,
            Instructions = _instructions,
            Temperature = _temperature,
            TopP = _topP,
            MaxOutputTokens = _maxOutputTokens,
            UseAudioInVideo = _useAudioInVideo,
            AutoRespond = _autoRespond,
            Model = _model,
            Done = true,
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task HandleResponseCreateAsync(OpenAIRealtimeEvent evt, CancellationToken cancellationToken)
    {
        if (!_autoRespond)
        {
            await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
            {
                Type = "response.created",
                SessionId = _sessionId,
                ResponseId = evt.ResponseId ?? $"resp_{Guid.NewGuid():N}",
                Status = "queued",
                Model = _model,
                Done = true,
            }, cancellationToken).ConfigureAwait(false);
            return;
        }

        await HandleAudioCommitAsync(cancellationToken).ConfigureAwait(false);
    }

    private void HandleConversationItemCreate(OpenAIRealtimeEvent evt)
    {
        var content = new List<OpenAIContentPart>();
        if (!string.IsNullOrWhiteSpace(evt.Text))
        {
            content.Add(new OpenAIContentPart { Type = "text", Text = evt.Text });
        }

        if (evt.Parts is not null)
        {
            content.AddRange(evt.Parts);
        }

        if (content.Count == 0)
        {
            return;
        }

        var itemId = string.IsNullOrWhiteSpace(evt.ItemId) ? $"item_{Guid.NewGuid():N}" : evt.ItemId;
        var previousItemId = _lastItemId;
        _conversation.Add(new OpenAIMessage
        {
            Role = evt.Role ?? "user",
            Parts = content,
            Content = FlattenParts(content),
        });
        _lastItemId = itemId;

        _ = _events.Writer.TryWrite(new OpenAIRealtimeEvent
        {
            Type = "conversation.item.created",
            SessionId = _sessionId,
            ItemId = itemId,
            PreviousItemId = previousItemId,
            Role = evt.Role ?? "user",
            Text = FlattenParts(content),
            Parts = content,
            Status = "completed",
            Done = true,
        });
    }

    private void HandleAudioAppend(OpenAIRealtimeEvent evt)
    {
        if (string.IsNullOrWhiteSpace(evt.AudioBase64))
        {
            return;
        }

        var bytes = Convert.FromBase64String(evt.AudioBase64);
        var format = evt.Format ?? "pcm16";
        float[] samples;
        switch (format.ToLowerInvariant())
        {
            case "wav":
                samples = DecodeWav(bytes, out var wavRate);
                _sampleRate = wavRate;
                break;
            case "pcm16":
                samples = WavCodec.DecodePcm16Samples(bytes);
                break;
            default:
                throw new NotSupportedException($"Unsupported realtime audio format: {format}");
        }

        if (!_speechStartedEmitted)
        {
            _speechStartedEmitted = true;
            _ = _events.Writer.TryWrite(new OpenAIRealtimeEvent
            {
                Type = "input_audio_buffer.speech_started",
                SessionId = _sessionId,
                SampleRate = _sampleRate,
                Done = true,
            });
        }

        _inputPcm.AddRange(samples);
        _ = _events.Writer.TryWrite(new OpenAIRealtimeEvent
        {
            Type = "input_audio_buffer.appended",
            SessionId = _sessionId,
            SampleRate = _sampleRate,
            Format = format,
            Done = true,
        });
    }

    private async Task HandleAudioCommitAsync(CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        CancellationTokenSource? generationCts = null;
        try
        {
            if (_inputPcm.Count == 0 && _conversation.Count == 0)
            {
                await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
                {
                    Type = "response.done",
                    SessionId = _sessionId,
                    Text = string.Empty,
                    Done = true,
                }, cancellationToken).ConfigureAwait(false);
                return;
            }

            generationCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            lock (_stateLock)
            {
                _generationCts?.Dispose();
                _generationCts = generationCts;
            }

            var userMessages = new List<OpenAIMessage>(_conversation);
            if (_inputPcm.Count > 0)
            {
                userMessages.Add(new OpenAIMessage
                {
                    Role = "user",
                    Parts =
                    [
                        new OpenAIContentPart
                        {
                            Type = "input_audio",
                            Audio = new OpenAIInputAudio
                            {
                                DataBase64 = Convert.ToBase64String(WavCodec.WritePcm16Wav(_inputPcm, _sampleRate)),
                                Format = "wav",
                            }
                    }
                ]
            });
            }

            if (!string.IsNullOrWhiteSpace(_instructions)
                && !userMessages.Any(message => string.Equals(message.Role, "system", StringComparison.OrdinalIgnoreCase)))
            {
                userMessages.Insert(0, new OpenAIMessage
                {
                    Role = "system",
                    Content = _instructions,
                });
            }

            await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
            {
                Type = "input_audio_buffer.committed",
                SessionId = _sessionId,
                SampleRate = _sampleRate,
                Done = true,
            }, generationCts.Token).ConfigureAwait(false);

            var response = await _service.CreateResponseAsync(new OpenAIResponseRequest
            {
                Model = _model,
                Input = userMessages,
                Voice = _voice,
                UseAudioInVideo = _useAudioInVideo,
                Modalities = _returnAudio
                    ? ["text", "audio"]
                    : ["text"],
                MaxOutputTokens = _maxOutputTokens ?? 256,
                Temperature = _temperature,
                TopP = _topP,
            }, generationCts.Token).ConfigureAwait(false);

            var output = response.Output.FirstOrDefault();
            if (output is null)
            {
                await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
                {
                    Type = "response.done",
                    SessionId = _sessionId,
                    Done = true,
                }, generationCts.Token).ConfigureAwait(false);
                return;
            }

            var responseId = $"resp_{Guid.NewGuid():N}";
            var previousAssistantItemId = _lastItemId;
            foreach (var warning in response.Warnings)
            {
                await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
                {
                    Type = "response.warning",
                    SessionId = _sessionId,
                    ResponseId = responseId,
                    Warning = warning,
                    Status = "degraded",
                    Model = response.Model,
                    Done = true,
                }, generationCts.Token).ConfigureAwait(false);
            }

            await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
            {
                Type = "response.created",
                SessionId = _sessionId,
                ResponseId = responseId,
                Status = "in_progress",
                Model = response.Model,
                Done = true,
            }, generationCts.Token).ConfigureAwait(false);

            var textChunks = SplitTextForStreaming(output.Content);
            if (textChunks.Count > 0)
            {
                var accumulated = new StringBuilder();
                foreach (var chunk in textChunks)
                {
                    accumulated.Append(chunk);
                    await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
                    {
                        Type = "response.output_text.delta",
                        SessionId = _sessionId,
                        ResponseId = responseId,
                        OutputIndex = 0,
                        ContentIndex = 0,
                        Text = chunk,
                        Delta = chunk,
                        Status = "in_progress",
                        Model = response.Model,
                    }, generationCts.Token).ConfigureAwait(false);
                }

                await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
                {
                    Type = "response.output_text.done",
                    SessionId = _sessionId,
                    ResponseId = responseId,
                    OutputIndex = 0,
                    ContentIndex = 0,
                    Text = accumulated.ToString(),
                    Status = "completed",
                    Model = response.Model,
                    Done = true,
                }, generationCts.Token).ConfigureAwait(false);
            }

            if (_returnAudio && !string.IsNullOrWhiteSpace(output.AudioBase64))
            {
                await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
                {
                    Type = "response.output_audio.delta",
                    SessionId = _sessionId,
                    ResponseId = responseId,
                    OutputIndex = 0,
                    ContentIndex = 1,
                    AudioBase64 = output.AudioBase64,
                    Format = output.AudioFormat ?? "wav",
                    Status = "in_progress",
                    Model = response.Model,
                }, generationCts.Token).ConfigureAwait(false);

                await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
                {
                    Type = "response.output_audio.done",
                    SessionId = _sessionId,
                    ResponseId = responseId,
                    OutputIndex = 0,
                    ContentIndex = 1,
                    AudioBase64 = output.AudioBase64,
                    Format = output.AudioFormat ?? "wav",
                    Status = "completed",
                    Model = response.Model,
                    Done = true,
                }, generationCts.Token).ConfigureAwait(false);
            }

            if (!string.IsNullOrWhiteSpace(output.Transcript))
            {
                _conversation.Add(new OpenAIMessage
                {
                    Role = "user",
                    Content = output.Transcript,
                });
            }

            PruneMultimodalConversationHistory();

            var assistantItemId = $"item_{Guid.NewGuid():N}";
            _lastItemId = assistantItemId;
            _conversation.Add(new OpenAIMessage
            {
                Role = "assistant",
                Content = output.Content,
            });

            _ = _events.Writer.TryWrite(new OpenAIRealtimeEvent
            {
                Type = "response.output_item.added",
                SessionId = _sessionId,
                ResponseId = responseId,
                ItemId = assistantItemId,
                PreviousItemId = previousAssistantItemId,
                Role = "assistant",
                Status = "completed",
                Model = response.Model,
                Done = true,
            });

            _ = _events.Writer.TryWrite(new OpenAIRealtimeEvent
            {
                Type = "response.content_part.added",
                SessionId = _sessionId,
                ResponseId = responseId,
                ItemId = assistantItemId,
                PreviousItemId = previousAssistantItemId,
                OutputIndex = 0,
                ContentIndex = 0,
                Text = output.Content,
                Delta = output.Content,
                Status = "completed",
                Done = true,
            });

            _ = _events.Writer.TryWrite(new OpenAIRealtimeEvent
            {
                Type = "conversation.item.created",
                SessionId = _sessionId,
                ItemId = assistantItemId,
                PreviousItemId = previousAssistantItemId,
                Role = "assistant",
                Text = output.Content,
                AudioBase64 = output.AudioBase64,
                Format = output.AudioFormat,
                Status = "completed",
                ResponseId = responseId,
                Model = response.Model,
                Done = true,
            });

            await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
            {
                Type = "response.done",
                SessionId = _sessionId,
                ResponseId = responseId,
                Status = "completed",
                Text = output.Content,
                AudioBase64 = output.AudioBase64,
                Format = output.AudioFormat,
                Model = response.Model,
                Done = true,
            }, generationCts.Token).ConfigureAwait(false);

            await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
            {
                Type = "response.completed",
                SessionId = _sessionId,
                ResponseId = responseId,
                Status = "completed",
                Model = response.Model,
                Done = true,
            }, generationCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
            {
                Type = "response.cancelled",
                SessionId = _sessionId,
                Done = true,
            }, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            await _events.Writer.WriteAsync(new OpenAIRealtimeEvent
            {
                Type = "error",
                SessionId = _sessionId,
                Text = ex.Message,
                Error = ex.Message,
                Status = "failed",
                Done = true,
            }, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _inputPcm.Clear();
            _speechStartedEmitted = false;
            lock (_stateLock)
            {
                if (ReferenceEquals(_generationCts, generationCts))
                {
                    _generationCts?.Dispose();
                    _generationCts = null;
                }
            }
            Qwen25OmniOpenAIService.CollectNativeTensorMemory();
            _gate.Release();
        }
    }

    private void CancelActiveGeneration()
    {
        lock (_stateLock)
        {
            _generationCts?.Cancel();
        }
    }

    private static float[] DecodeWav(byte[] bytes, out int sampleRate)
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"qwen_realtime_{Guid.NewGuid():N}.wav");
        File.WriteAllBytes(tempPath, bytes);
        try
        {
            return WavCodec.ReadMonoSamples(tempPath, out sampleRate);
        }
        finally
        {
            try
            {
                File.Delete(tempPath);
            }
            catch
            {
            }
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _events.Writer.TryComplete();
        _gate.Dispose();
        await Task.CompletedTask;
    }

    private void PruneMultimodalConversationHistory()
    {
        for (var i = 0; i < _conversation.Count; i++)
        {
            var message = _conversation[i];
            if (message.Parts is null || !message.Parts.Any(IsNonTextPart))
            {
                continue;
            }

            var text = !string.IsNullOrWhiteSpace(message.Content)
                ? message.Content!
                : FlattenParts(message.Parts);
            if (string.IsNullOrWhiteSpace(text))
            {
                text = "[previous multimodal input omitted]";
            }

            _conversation[i] = new OpenAIMessage
            {
                Role = message.Role,
                Content = text,
            };
        }
    }

    private static bool IsNonTextPart(OpenAIContentPart part)
        => !string.Equals(part.Type, "text", StringComparison.OrdinalIgnoreCase);

    private static string FlattenParts(IEnumerable<OpenAIContentPart> parts)
    {
        var builder = new StringBuilder();
        foreach (var part in parts)
        {
            if (string.Equals(part.Type, "text", StringComparison.OrdinalIgnoreCase) && !string.IsNullOrWhiteSpace(part.Text))
            {
                builder.Append(part.Text);
            }
        }

        return builder.ToString();
    }

    private static List<string> SplitTextForStreaming(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return [];
        }

        var result = new List<string>();
        var current = new StringBuilder();
        foreach (var ch in text)
        {
            current.Append(ch);
            if (ch is '.' or '!' or '?' or '\n')
            {
                result.Add(current.ToString());
                current.Clear();
            }
        }

        if (current.Length > 0)
        {
            result.Add(current.ToString());
        }

        return result;
    }
}
