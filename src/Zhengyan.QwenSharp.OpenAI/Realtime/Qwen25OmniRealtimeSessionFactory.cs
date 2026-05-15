using Zhengyan.QwenSharp.OpenAI;

namespace Zhengyan.QwenSharp.OpenAI.Realtime;

public sealed class Qwen25OmniRealtimeSessionFactory : IOpenAIRealtimeSessionFactory
{
    private readonly Qwen25OmniOpenAIService _service;

    public Qwen25OmniRealtimeSessionFactory(Qwen25OmniOpenAIService service)
    {
        _service = service;
    }

    public ValueTask<IOpenAIRealtimeSession> CreateAsync(OpenAIRealtimeSessionOptions options, CancellationToken cancellationToken = default)
    {
        IOpenAIRealtimeSession session = new Qwen25OmniRealtimeSession(_service, options);
        return ValueTask.FromResult(session);
    }
}
