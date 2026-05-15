using TorchSharp;
using Zhengyan.QwenSharp.Core;
using Zhengyan.QwenSharp.Generation;
using Zhengyan.QwenSharp.Models;
using Zhengyan.QwenSharp.Tokenizers;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.OpenAI;

public sealed class QwenTextOpenAIService : IOpenAIChatCompletionsService, IOpenAIResponsesService, IDisposable
{
    private readonly Qwen2Tokenizer _tokenizer;
    private readonly ICausalLM _model;
    private readonly Device _device;
    private readonly ScalarType? _modelDtype;
    private readonly string _modelName;

    public QwenTextOpenAIService(
        string modelDirectory,
        string? device = null,
        string? dtype = null,
        string? modelName = null)
    {
        _tokenizer = Qwen2Tokenizer.FromDirectory(modelDirectory);
        _model = QwenModelLoader.LoadCausalLM(modelDirectory, out var detectedModelType);
        _modelName = string.IsNullOrWhiteSpace(modelName) ? detectedModelType : modelName;
        _device = TorchHelper.ParseDevice(device);
        _modelDtype = TorchHelper.ResolveModelDtype(dtype, _device, null);
        TorchHelper.InitializeRuntime(_device);

        var module = (torch.nn.Module)_model;
        if (_modelDtype is not null)
        {
            module.to(_modelDtype.Value);
        }

        module.to(_device);
        Console.WriteLine($"{_modelName} model loaded on {_device} with dtype {_modelDtype?.ToString() ?? "default"}.");
    }

    public Task<OpenAIChatCompletionResponse> CreateCompletionAsync(OpenAIChatCompletionRequest request, CancellationToken cancellationToken = default)
    {
        var reply = GenerateText(request.Messages, request.Temperature, request.TopP, request.MaxTokens);
        return Task.FromResult(new OpenAIChatCompletionResponse
        {
            Model = string.IsNullOrWhiteSpace(request.Model) ? _modelName : request.Model,
            Choices =
            [
                new OpenAIChatCompletionChoice
                {
                    Index = 0,
                    Message = new OpenAIMessage { Role = "assistant", Content = reply },
                    FinishReason = "stop",
                }
            ],
        });
    }

    public Task<OpenAIResponseResponse> CreateResponseAsync(OpenAIResponseRequest request, CancellationToken cancellationToken = default)
    {
        var messages = ApplyInstructions(request.Input, request.Instructions);
        var reply = GenerateText(messages, request.Temperature, request.TopP, request.MaxOutputTokens);
        return Task.FromResult(new OpenAIResponseResponse
        {
            Model = string.IsNullOrWhiteSpace(request.Model) ? _modelName : request.Model,
            Output =
            [
                new OpenAIResponseOutput
                {
                    Content = reply,
                }
            ],
        });
    }

    private string GenerateText(IReadOnlyList<OpenAIMessage> messages, float? temperature, float? topP, int? maxTokens)
    {
        var chatMessages = messages
            .Select(message => new ChatMessage(NormalizeRole(message.Role), FlattenMessage(message)))
            .Where(message => !string.IsNullOrWhiteSpace(message.Content))
            .ToList();
        var promptIds = _tokenizer.EncodeChatTemplate(chatMessages);
        using var inputIds = tensor(promptIds, dtype: ScalarType.Int64, device: _device).unsqueeze(0);
        var config = new GenerationConfig
        {
            MaxNewTokens = Math.Clamp(maxTokens ?? 256, 1, 1024),
            Temperature = temperature ?? 0.2f,
            TopP = topP ?? 0.9f,
            TopK = 40,
            DoSample = (temperature ?? 0.2f) > 0,
        };

        var tokens = new List<int>();
        foreach (var token in TextGenerator.Generate(_model, inputIds, config, _tokenizer.EosTokenId ?? -1, _tokenizer.ImEndId))
        {
            tokens.Add(token);
        }

        return _tokenizer.Decode(tokens, skipSpecialTokens: true).Trim();
    }

    private static IReadOnlyList<OpenAIMessage> ApplyInstructions(IReadOnlyList<OpenAIMessage> messages, string? instructions)
    {
        if (string.IsNullOrWhiteSpace(instructions))
        {
            return messages;
        }

        var list = messages.ToList();
        list.Insert(0, new OpenAIMessage { Role = "system", Content = instructions });
        return list;
    }

    private static string FlattenMessage(OpenAIMessage message)
    {
        if (!string.IsNullOrWhiteSpace(message.Content))
        {
            return message.Content!;
        }

        if (message.Parts is null)
        {
            return string.Empty;
        }

        return string.Concat(message.Parts
            .Where(part => string.Equals(part.Type, "text", StringComparison.OrdinalIgnoreCase))
            .Select(part => part.Text));
    }

    private static string NormalizeRole(string role)
        => string.Equals(role, "system", StringComparison.OrdinalIgnoreCase)
            || string.Equals(role, "assistant", StringComparison.OrdinalIgnoreCase)
            || string.Equals(role, "tool", StringComparison.OrdinalIgnoreCase)
            ? role.ToLowerInvariant()
            : "user";

    public void Dispose()
    {
        (_model as IDisposable)?.Dispose();
    }
}
