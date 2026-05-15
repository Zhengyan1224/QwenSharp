using System.Collections.Generic;

namespace Zhengyan.QwenSharp.Tokenizers;

public interface ITokenizer
{
    /// <summary>
    /// Encodes a text string into an array of token IDs.
    /// </summary>
    int[] Encode(string text, bool addSpecialTokens = true);
    
    /// <summary>
    /// Decodes an array of token IDs back into a string.
    /// </summary>
    string Decode(IReadOnlyList<int> tokenIds, bool skipSpecialTokens = true);
    
    /// <summary>
    /// Encodes a chat conversation into token IDs following the model's chat template.
    /// </summary>
    int[] EncodeChatTemplate(IEnumerable<ChatMessage> messages, bool enableThinking = false);
    
    int VocabSize { get; }
    int? BosTokenId { get; }
    int? EosTokenId { get; }
    int? PadTokenId { get; }
}

public class ChatMessage
{
    public string Role { get; set; } = "";
    public string Content { get; set; } = "";

    public ChatMessage(string role, string content)
    {
        Role = role;
        Content = content;
    }
}
