using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.ML.Tokenizers;

namespace Zhengyan.QwenSharp.Tokenizers;

public class Qwen2Tokenizer : ITokenizer
{
    private readonly Tokenizer _tokenizer;
    private readonly Dictionary<string, int> _addedTokens;
    private readonly Dictionary<int, string> _addedTokensDecoder;

    public int VocabSize { get; }
    public int? BosTokenId { get; }
    public int? EosTokenId { get; }
    public int? PadTokenId { get; }

    public int ImStartId { get; }
    public int ImEndId { get; }

    public const string ImStart = "<|im_start|>";
    public const string ImEnd = "<|im_end|>";

    private Qwen2Tokenizer(
        Tokenizer tokenizer,
        Dictionary<string, int> addedTokens,
        int vocabSize,
        int? bosTokenId,
        int? eosTokenId,
        int? padTokenId)
    {
        _tokenizer = tokenizer;
        _addedTokens = addedTokens ?? new();
        _addedTokensDecoder = new Dictionary<int, string>();

        foreach (var kvp in _addedTokens)
            _addedTokensDecoder[kvp.Value] = kvp.Key;

        VocabSize = vocabSize;
        BosTokenId = bosTokenId;
        EosTokenId = eosTokenId;
        PadTokenId = padTokenId;

        _addedTokens.TryGetValue(ImStart, out var imStartId);
        ImStartId = imStartId;

        _addedTokens.TryGetValue(ImEnd, out var imEndId);
        ImEndId = imEndId;
    }

    public static Qwen2Tokenizer FromDirectory(string modelDirectory)
    {
        var vocabPath = Path.Combine(modelDirectory, "vocab.json");
        var mergesPath = Path.Combine(modelDirectory, "merges.txt");

        // Try loading Tiktoken format first (Qwen's default)
        var tiktokenFiles = Directory.GetFiles(modelDirectory, "*.tiktoken");
        Tokenizer? tokenizerBase = null;

        if (tiktokenFiles.Length > 0)
        {
            // The vocab is loaded later with special tokens appended
        }
        else
        {
            if (!File.Exists(vocabPath) || !File.Exists(mergesPath))
                throw new FileNotFoundException($"BPE vocab.json and merges.txt, or a .tiktoken file, are required in {modelDirectory}");
        }

        // Load added tokens
        var addedTokens = new Dictionary<string, int>();
        var addedTokensPath = Path.Combine(modelDirectory, "added_tokens.json");
        if (File.Exists(addedTokensPath))
        {
            try
            {
                var json = File.ReadAllText(addedTokensPath);
                addedTokens = JsonSerializer.Deserialize<Dictionary<string, int>>(json) ?? new();
            }
            catch { /* Ignore parse errors */ }
        }

        // Always load added tokens from tokenizer.json - this is the authoritative source
        // for all special tokens including model-specific ones like <think> (Qwen3) 
        // and correctly-numbered im_start/im_end for Qwen3.5 (vocab 248320)
        var tokenizerJsonPath = Path.Combine(modelDirectory, "tokenizer.json");
        if (File.Exists(tokenizerJsonPath))
        {
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
                if (doc.RootElement.TryGetProperty("added_tokens", out var added))
                {
                    foreach (var token in added.EnumerateArray())
                    {
                        var content = token.GetProperty("content").GetString()!;
                        var id = token.GetProperty("id").GetInt32();
                        addedTokens[content] = id;  // tokenizer.json always wins
                    }
                }
            }
            catch { }
        }

        // Qwen3.5 models store special tokens inside tokenizer_config.json's added_tokens_decoder
        var tokenizerConfigPath = Path.Combine(modelDirectory, "tokenizer_config.json");
        if (File.Exists(tokenizerConfigPath))
        {
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerConfigPath));
                if (doc.RootElement.TryGetProperty("added_tokens_decoder", out var decoder))
                {
                    foreach (var tokenProp in decoder.EnumerateObject())
                    {
                        if (int.TryParse(tokenProp.Name, out var id))
                        {
                            var content = tokenProp.Value.GetProperty("content").GetString()!;
                            addedTokens[content] = id;
                        }
                    }
                }
            }
            catch { }
        }

        // Qwen defaults
        if (!addedTokens.ContainsKey("<|endoftext|>")) addedTokens["<|endoftext|>"] = 151643;
        if (!addedTokens.ContainsKey(ImStart)) addedTokens[ImStart] = 151644;
        if (!addedTokens.ContainsKey(ImEnd)) addedTokens[ImEnd] = 151645;

        // Model config
        int vocabSize = 151936;
        int? bosTokenId = null;
        int? eosTokenId = 151643;
        int? padTokenId = 151643;

        var configPath = Path.Combine(modelDirectory, "config.json");
        if (File.Exists(configPath))
        {
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
                
                // For multimodal Qwen3.5 models, token IDs are nested in text_config
                var root = doc.RootElement;
                JsonElement configElem = root;
                if (root.TryGetProperty("text_config", out var textCfg))
                    configElem = textCfg;
                
                if (configElem.TryGetProperty("vocab_size", out var v)) vocabSize = v.GetInt32();
                if (configElem.TryGetProperty("bos_token_id", out var b)) bosTokenId = b.GetInt32();
                if (configElem.TryGetProperty("pad_token_id", out var p)) padTokenId = p.GetInt32();
                
                // eos_token_id can be an array or int
                if (configElem.TryGetProperty("eos_token_id", out var e))
                {
                    if (e.ValueKind == JsonValueKind.Number)
                        eosTokenId = e.GetInt32();
                    else if (e.ValueKind == JsonValueKind.Array && e.GetArrayLength() > 0)
                        eosTokenId = e[0].GetInt32();
                }
            }
            catch { }
        }

        if (tokenizerBase == null && tiktokenFiles.Length > 0)
        {
            tokenizerBase = BPETokenizer.CreateTiktoken(tiktokenFiles[0], addedTokens);
        }

        if (tokenizerBase == null && File.Exists(vocabPath))
        {
            tokenizerBase = BPETokenizer.CreateTiktokenFromQwenVocab(vocabPath, addedTokens);
        }

        if (tokenizerBase == null)
        {
            tokenizerBase = BPETokenizer.CreateQwenBpe(vocabPath, mergesPath, addedTokens);
        }

        return new Qwen2Tokenizer(tokenizerBase!, addedTokens, vocabSize, bosTokenId, eosTokenId, padTokenId);
    }

    public int[] Encode(string text, bool addSpecialTokens = true)
    {
        if (string.IsNullOrEmpty(text)) return [];

        // If no added tokens or not needed, fast path
        if (!addSpecialTokens || _addedTokens.Count == 0)
        {
            return _tokenizer.EncodeToIds(text).ToArray();
        }

        // Segment the text: split around added special tokens, encode each part
        // This correctly handles <|im_start|>, <think>, </think>, etc.
        var result = new List<int>();
        int pos = 0;
        var textSpan = text.AsSpan();

        while (pos < text.Length)
        {
            // Try to find the earliest occurring special token from current pos onwards
            int bestMatchPos = -1;
            int bestMatchLen = -1;
            string? bestMatchKey = null;

            foreach (var kvp in _addedTokens)
            {
                var token = kvp.Key;
                var idx = text.IndexOf(token, pos, StringComparison.Ordinal);
                if (idx >= 0 && (bestMatchPos < 0 || idx < bestMatchPos || (idx == bestMatchPos && token.Length > bestMatchLen)))
                {
                    bestMatchPos = idx;
                    bestMatchLen = token.Length;
                    bestMatchKey = token;
                }
            }

            if (bestMatchPos < 0)
            {
                // No more special tokens found, encode the rest as BPE
                var rest = text[pos..];
                if (!string.IsNullOrEmpty(rest))
                    result.AddRange(_tokenizer.EncodeToIds(rest));
                break;
            }

            // Encode text before the special token using BPE
            if (bestMatchPos > pos)
            {
                var before = text[pos..bestMatchPos];
                result.AddRange(_tokenizer.EncodeToIds(before));
            }

            // Add the special token's ID
            result.Add(_addedTokens[bestMatchKey!]);
            pos = bestMatchPos + bestMatchLen;
        }

        return result.ToArray();
    }

    // GPT-2 / tiktoken byte-level encoding: maps each raw byte to a specific Unicode character.
    // This is the REVERSE direction of `bytes_to_unicode()` in the original OpenAI code.
    // We need this to convert the token text back to real UTF-8 bytes.
    private static readonly Dictionary<char, byte> ByteDecoderMap = BuildByteDecoder();

    private static Dictionary<char, byte> BuildByteDecoder()
    {
        // Replicate `bytes_to_unicode()` from OpenAI gpt-2/tiktoken:
        // printable ASCII + Latin extended that don't need escaping are mapped to themselves.
        // The remaining 256-n bytes are mapped to Unicode starting at 256.
        var bs = new List<int>();
        for (int i = '!'; i <= '~'; i++) bs.Add(i);      // 33-126
        for (int i = 0xA1; i <= 0xAC; i++) bs.Add(i);    // 161-172
        for (int i = 0xAE; i <= 0xFF; i++) bs.Add(i);    // 174-255

        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }

        var decoder = new Dictionary<char, byte>(256);
        for (int i = 0; i < bs.Count; i++)
            decoder[(char)cs[i]] = (byte)bs[i];
        return decoder;
    }

    private static string ByteDecodeTiktoken(string raw)
    {
        var bytes = new byte[raw.Length];
        for (int i = 0; i < raw.Length; i++)
        {
            if (ByteDecoderMap.TryGetValue(raw[i], out var b))
                bytes[i] = b;
            else
                bytes[i] = (byte)raw[i]; // fallback for regular chars 
        }
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    private static readonly Dictionary<byte, char> ByteEncoderMap = BuildByteEncoder();

    private static Dictionary<byte, char> BuildByteEncoder()
    {
        var bs = new List<int>();
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        for (int i = 0xA1; i <= 0xAC; i++) bs.Add(i);
        for (int i = 0xAE; i <= 0xFF; i++) bs.Add(i);

        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }

        var encoder = new Dictionary<byte, char>(256);
        for (int i = 0; i < bs.Count; i++)
            encoder[(byte)bs[i]] = (char)cs[i];
        return encoder;
    }

    private static string ByteEncodeTiktoken(string raw)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(raw);
        var sb = new System.Text.StringBuilder(bytes.Length);
        foreach (var b in bytes)
        {
            if (ByteEncoderMap.TryGetValue(b, out var c))
                sb.Append(c);
            else
                sb.Append((char)b); // Fallback
        }
        return sb.ToString();
    }

    public string Decode(IReadOnlyList<int> tokenIds, bool skipSpecialTokens = true)
    {
        if (tokenIds == null || tokenIds.Count == 0) return string.Empty;

        var sb = new StringBuilder();
        var tempIds = new List<int>();

        foreach (var id in tokenIds)
        {
            if (_addedTokensDecoder.TryGetValue(id, out var specialToken))
            {
                // Decode buffered normal tokens
                if (tempIds.Count > 0)
                {
                    sb.Append(_tokenizer.Decode(tempIds.ToArray()) ?? string.Empty);
                    tempIds.Clear();
                }

                if (!skipSpecialTokens)
                {
                    sb.Append(specialToken);
                }
            }
            else
            {
                tempIds.Add(id);
            }
        }

        if (tempIds.Count > 0)
        {
            sb.Append(_tokenizer.Decode(tempIds.ToArray()) ?? string.Empty);
        }

        return sb.ToString();
    }

    public int[] EncodeChatTemplate(IEnumerable<ChatMessage> messages, bool enableThinking = false)
    {
        var ids = new List<int>();
        var msgList = messages.ToList();
        
        // Qwen models generally require a system prompt to function properly as an assistant
        if (msgList.Count == 0 || msgList[0].Role != "system")
        {
            msgList.Insert(0, new ChatMessage("system", "You are a helpful assistant."));
        }

        foreach (var msg in msgList)
        {
            ids.Add(ImStartId);
            ids.AddRange(Encode(msg.Role, addSpecialTokens: false));
            ids.AddRange(Encode("\n", addSpecialTokens: false));
            ids.AddRange(Encode(msg.Content, addSpecialTokens: false));
            ids.Add(ImEndId);
            ids.AddRange(Encode("\n", addSpecialTokens: false));
        }
        
        // Add generation prompt
        ids.Add(ImStartId);
        ids.AddRange(Encode("assistant\n", addSpecialTokens: false));
        
        // When enableThinking=false, pre-fill an empty think block to skip Qwen3 reasoning mode.
        // This is equivalent to `enable_thinking=False` in the official Jinja template and prevents
        // the model from generating unrelated internal reasoning before responding.
        if (!enableThinking)
        {
            ids.AddRange(Encode("<think>", addSpecialTokens: true));
            ids.AddRange(Encode("\n\n", addSpecialTokens: false));
            ids.AddRange(Encode("</think>", addSpecialTokens: true));
            ids.AddRange(Encode("\n\n", addSpecialTokens: false));
        }

        return ids.ToArray();
    }
}
