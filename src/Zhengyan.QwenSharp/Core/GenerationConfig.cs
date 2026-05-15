namespace Zhengyan.QwenSharp.Core;

/// <summary>
/// Configuration for text generation strategies.
/// Mirrors HuggingFace GenerationConfig.
/// </summary>
public class GenerationConfig
{
    /// <summary>Maximum number of new tokens to generate.</summary>
    public int MaxNewTokens { get; set; } = 512;

    /// <summary>Maximum total sequence length (prompt + generated).</summary>
    public int? MaxLength { get; set; }

    /// <summary>Temperature for sampling. 1.0 = no change, &lt;1.0 = sharper, &gt;1.0 = flatter.</summary>
    public float Temperature { get; set; } = 1.0f;

    /// <summary>Top-p (nucleus) sampling. 0.0 = disabled, 1.0 = no filtering.</summary>
    public float TopP { get; set; } = 1.0f;

    /// <summary>Top-k sampling. 0 = disabled.</summary>
    public int TopK { get; set; } = 50;

    /// <summary>Number of beams for beam search. 1 = greedy/sampling.</summary>
    public int NumBeams { get; set; } = 1;

    /// <summary>Whether to use sampling. False = greedy (or beam search).</summary>
    public bool DoSample { get; set; } = false;

    /// <summary>Repetition penalty. 1.0 = no penalty.</summary>
    public float RepetitionPenalty { get; set; } = 1.0f;

    /// <summary>EOS token IDs to stop generation.</summary>
    public int[]? EosTokenId { get; set; }

    /// <summary>PAD token ID.</summary>
    public int? PadTokenId { get; set; }

    /// <summary>Whether to use KV cache for faster inference.</summary>
    public bool UseCache { get; set; } = true;

    /// <summary>
    /// Creates a greedy decoding config (deterministic, fastest).
    /// </summary>
    public static GenerationConfig Greedy(int maxNewTokens = 512) => new()
    {
        MaxNewTokens = maxNewTokens,
        DoSample = false,
        Temperature = 1.0f,
    };

    /// <summary>
    /// Creates a sampling config with typical chat settings.
    /// </summary>
    public static GenerationConfig ChatSampling(int maxNewTokens = 2048) => new()
    {
        MaxNewTokens = maxNewTokens,
        DoSample = true,
        Temperature = 0.7f,
        TopP = 0.9f,
        TopK = 50,
    };
}
