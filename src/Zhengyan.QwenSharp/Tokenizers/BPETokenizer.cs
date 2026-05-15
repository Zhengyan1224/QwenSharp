using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace Zhengyan.QwenSharp.Tokenizers;

/// <summary>
/// A Tokenizer that loads from a Tiktoken BPE dictionary (e.g., qwen.tiktoken).
/// This wraps Microsoft.ML.Tokenizers.TiktokenTokenizer for seamless integration.
/// </summary>
public static class BPETokenizer
{
    private static readonly Regex QwenPreTokenizerRegex = new(
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    public static Tokenizer CreateTiktoken(string tiktokenFilePath, IReadOnlyDictionary<string, int>? specialTokens = null)
    {
        if (!File.Exists(tiktokenFilePath))
        {
            throw new FileNotFoundException($"The requested tiktoken file was not found: {tiktokenFilePath}");
        }

        // Qwen models generally do not require complex pre-tokenization for generation contexts.
        // We use TiktokenTokenizer directly with the vocab file.
        var dictKeys = new Dictionary<string, int>();
        if (specialTokens != null)
        {
            foreach (var kvp in specialTokens)
            {
                dictKeys[kvp.Key] = kvp.Value;
            }
        }

        PreTokenizer? preTokenizer = null;

        var tokenizer = TiktokenTokenizer.Create(
            vocabFilePath: tiktokenFilePath,
            preTokenizer: preTokenizer,
            normalizer: null,
            specialTokens: null, // Let Qwen2Tokenizer handle special tokens
            cacheSize: 8192
        );

        return tokenizer;
    }

    public static Tokenizer CreateTiktokenFromQwenVocab(string vocabFilePath, IReadOnlyDictionary<string, int>? specialTokens = null)
    {
        if (!File.Exists(vocabFilePath))
        {
            throw new FileNotFoundException($"The requested vocab file was not found: {vocabFilePath}");
        }

        var tiktokenContent = BuildTiktokenContentFromVocab(vocabFilePath);
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(tiktokenContent));
        return TiktokenTokenizer.Create(
            stream,
            preTokenizer: null,
            normalizer: null,
            specialTokens: null,
            cacheSize: 8192);
    }

    public static Tokenizer CreateQwenBpe(string vocabFilePath, string mergesFilePath, IReadOnlyDictionary<string, int>? specialTokens = null)
    {
        if (!File.Exists(vocabFilePath))
        {
            throw new FileNotFoundException($"The requested vocab file was not found: {vocabFilePath}");
        }

        if (!File.Exists(mergesFilePath))
        {
            throw new FileNotFoundException($"The requested merges file was not found: {mergesFilePath}");
        }

        var preTokenizer = new RegexPreTokenizer(QwenPreTokenizerRegex, specialTokens ?? new Dictionary<string, int>());

        return BpeTokenizer.Create(
            vocabFilePath,
            mergesFilePath,
            preTokenizer,
            normalizer: null,
            specialTokens: null,
            unknownToken: null,
            continuingSubwordPrefix: null,
            endOfWordSuffix: null,
            true);
    }

    private static string BuildTiktokenContentFromVocab(string vocabFilePath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(vocabFilePath));
        var entries = doc.RootElement
            .EnumerateObject()
            .Select(prop => (Token: prop.Name, Rank: prop.Value.GetInt32()))
            .OrderBy(item => item.Rank);

        var sb = new StringBuilder();
        foreach (var (token, rank) in entries)
        {
            var bytes = DecodeVocabTokenToBytes(token);
            sb.Append(Convert.ToBase64String(bytes));
            sb.Append(' ');
            sb.Append(rank);
            sb.Append('\n');
        }

        return sb.ToString();
    }

    private static byte[] DecodeVocabTokenToBytes(string token)
    {
        var bytes = new byte[token.Length];
        for (int i = 0; i < token.Length; i++)
        {
            if (ByteDecoderMap.TryGetValue(token[i], out var value))
            {
                bytes[i] = value;
            }
            else
            {
                bytes[i] = (byte)token[i];
            }
        }

        return bytes;
    }

    private static readonly Dictionary<char, byte> ByteDecoderMap = BuildByteDecoder();

    private static Dictionary<char, byte> BuildByteDecoder()
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

        var decoder = new Dictionary<char, byte>(256);
        for (int i = 0; i < bs.Count; i++)
        {
            decoder[(char)cs[i]] = (byte)bs[i];
        }

        return decoder;
    }
}

/*
public class QwenRegexPreTokenizer : PreTokenizer
{
    // The canonical Qwen tiktoken pat.
    private static readonly System.Text.RegularExpressions.Regex _regex = new(
        @"(?i:'s|'t||'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        System.Text.RegularExpressions.RegexOptions.Compiled);

    public override IEnumerable<(int, int)> PreTokenize(string text)
    {
        var matches = _regex.Matches(text);
        foreach (System.Text.RegularExpressions.Match match in matches)
        {
            yield return (match.Index, match.Length);
        }
    }
}
*/
