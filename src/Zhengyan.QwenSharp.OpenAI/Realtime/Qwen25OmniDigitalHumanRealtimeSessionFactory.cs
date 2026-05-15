using Zhengyan.QwenSharp.OpenAI;

namespace Zhengyan.QwenSharp.OpenAI.Realtime;

public sealed class Qwen25OmniDigitalHumanRealtimeSessionFactory : IOpenAIRealtimeSessionFactory
{
    private readonly Qwen25OmniOpenAIService _service;

    public Qwen25OmniDigitalHumanRealtimeSessionFactory(Qwen25OmniOpenAIService service)
    {
        _service = service;
    }

    public ValueTask<IOpenAIRealtimeSession> CreateAsync(OpenAIRealtimeSessionOptions options, CancellationToken cancellationToken = default)
    {
        IOpenAIRealtimeSession session = new Qwen25OmniDigitalHumanRealtimeSession(_service, options);
        return ValueTask.FromResult(session);
    }
}
