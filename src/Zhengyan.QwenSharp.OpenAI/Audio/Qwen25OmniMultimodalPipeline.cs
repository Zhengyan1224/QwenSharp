using System.Text;
using Zhengyan.QwenSharp.OpenAI;

namespace Zhengyan.QwenSharp.OpenAI.Audio;

internal sealed record Qwen25OmniMultimodalPipeline(
    Qwen25OmniMmInfo MmInfo,
    IReadOnlyList<OpenAIMessage> PreparedMessages,
    string SystemPrompt,
    bool WantsAudio,
    bool UseAudioInVideo) : IDisposable
{
    public void Dispose()
    {
        MmInfo.Dispose();
    }
}

internal static class Qwen25OmniMultimodalPipelineProcessor
{
    public static async Task<Qwen25OmniMultimodalPipeline> BuildAsync(
        IReadOnlyList<OpenAIMessage> messages,
        string systemPrompt,
        bool wantsAudio,
        bool useAudioInVideo,
        CancellationToken cancellationToken = default)
    {
        var preparedMessages = PrepareMessages(messages, wantsAudio);
        var mmInfo = await Qwen25OmniMmInfoProcessor.ProcessMmInfoAsync(preparedMessages, systemPrompt, useAudioInVideo, cancellationToken).ConfigureAwait(false);
        return new Qwen25OmniMultimodalPipeline(mmInfo, preparedMessages, systemPrompt, wantsAudio, useAudioInVideo);
    }

    public static Qwen25OmniMultimodalPipeline Build(
        IReadOnlyList<OpenAIMessage> messages,
        string systemPrompt,
        bool wantsAudio,
        bool useAudioInVideo)
    {
        var preparedMessages = PrepareMessages(messages, wantsAudio);
        var mmInfo = Qwen25OmniMmInfoProcessor.ProcessMmInfo(preparedMessages, systemPrompt, useAudioInVideo);
        return new Qwen25OmniMultimodalPipeline(mmInfo, preparedMessages, systemPrompt, wantsAudio, useAudioInVideo);
    }

    private static IReadOnlyList<OpenAIMessage> PrepareMessages(IReadOnlyList<OpenAIMessage> messages, bool wantsAudio)
    {
        var list = messages.ToList();
        if (wantsAudio)
        {
            const string AudioOutputSystemPrompt = "System capability note: you can perceive auditory and visual inputs and generate both text and speech. Keep any existing role, persona, and style instructions as the highest priority.";
            var hasAudioPrompt = list.Count > 0
                && string.Equals(list[0].Role, "system", StringComparison.OrdinalIgnoreCase)
                && FlattenMessage(list[0]).Contains(AudioOutputSystemPrompt, StringComparison.Ordinal);

            if (!hasAudioPrompt)
            {
                if (list.Count > 0 && string.Equals(list[0].Role, "system", StringComparison.OrdinalIgnoreCase))
                {
                    var existing = FlattenMessage(list[0]);
                    list[0] = list[0] with
                    {
                        Content = string.IsNullOrWhiteSpace(existing)
                            ? AudioOutputSystemPrompt
                            : $"{existing}\n\n{AudioOutputSystemPrompt}",
                    };
                }
                else
                {
                    list.Insert(0, new OpenAIMessage
                    {
                        Role = "system",
                        Content = AudioOutputSystemPrompt,
                    });
                }
            }
        }

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

        var text = new StringBuilder();
        foreach (var part in message.Parts)
        {
            if (string.Equals(part.Type, "text", StringComparison.OrdinalIgnoreCase) && !string.IsNullOrWhiteSpace(part.Text))
            {
                text.Append(part.Text);
            }
        }

        return text.ToString();
    }
}
