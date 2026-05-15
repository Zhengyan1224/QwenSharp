using SherpaOnnx;

namespace Zhengyan.QwenSharp.Omni.Audio;

public enum SherpaOnnxTtsModelKind
{
    Vits,
    Matcha,
}

public sealed class SherpaOnnxAudioSynthesizerOptions
{
    public required string ModelPath { get; init; }

    public required string TokensPath { get; init; }

    public SherpaOnnxTtsModelKind ModelKind { get; init; } = SherpaOnnxTtsModelKind.Vits;

    public string? LexiconPath { get; init; }

    public string? DataDirectory { get; init; }

    public string? DictDirectory { get; init; }

    public string? VocoderPath { get; init; }

    public string? RuleFars { get; init; }

    public string? RuleFsts { get; init; }

    public string Provider { get; init; } = "cpu";

    public int Threads { get; init; } = Math.Max(1, Environment.ProcessorCount / 2);

    public float Speed { get; init; } = 1.0f;

    public int SpeakerId { get; init; }

    public float NoiseScale { get; init; } = 0.667f;

    public float NoiseScaleW { get; init; } = 0.8f;

    public float LengthScale { get; init; } = 1.0f;
}

public sealed class SherpaOnnxAudioSynthesizer : IQwen25OmniAudioSynthesizer
{
    private readonly SherpaOnnxAudioSynthesizerOptions _options;
    private readonly Lazy<OfflineTts> _tts;
    private bool _disposed;

    public SherpaOnnxAudioSynthesizer(SherpaOnnxAudioSynthesizerOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentException.ThrowIfNullOrWhiteSpace(options.ModelPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(options.TokensPath);
        ValidateFile(options.ModelPath, "--tts-model");
        ValidateFile(options.TokensPath, "--tts-tokens");
        ValidateOptionalFile(options.LexiconPath, "--tts-lexicon");
        ValidateOptionalFile(options.VocoderPath, "--tts-vocoder");

        _options = options;
        _tts = new Lazy<OfflineTts>(CreateTts, LazyThreadSafetyMode.ExecutionAndPublication);
    }

    public string Name => $"SherpaOnnx.{_options.ModelKind}";

    public string SynthesizeToWavBase64(string text, string? voice = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (string.IsNullOrWhiteSpace(text))
        {
            text = ".";
        }

        var generated = _tts.Value.Generate(text, _options.Speed, _options.SpeakerId);
        try
        {
            var wavBytes = WavCodec.WritePcm16Wav(generated.Samples, generated.SampleRate);
            return Convert.ToBase64String(wavBytes);
        }
        finally
        {
            generated.Dispose();
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_tts.IsValueCreated)
        {
            _tts.Value.Dispose();
        }
    }

    private OfflineTts CreateTts()
    {
        var modelConfig = new OfflineTtsModelConfig
        {
            Provider = _options.Provider,
            NumThreads = _options.Threads,
        };

        switch (_options.ModelKind)
        {
            case SherpaOnnxTtsModelKind.Vits:
                modelConfig.Vits = new OfflineTtsVitsModelConfig
                {
                    Model = _options.ModelPath,
                    Tokens = _options.TokensPath,
                    Lexicon = _options.LexiconPath ?? ResolveSiblingFile(_options.ModelPath, "lexicon.txt") ?? string.Empty,
                    DataDir = _options.DataDirectory ?? ResolveSiblingDirectory(_options.ModelPath, "espeak-ng-data") ?? string.Empty,
                    DictDir = _options.DictDirectory ?? ResolveSiblingDirectory(_options.ModelPath, "dict") ?? string.Empty,
                    NoiseScale = _options.NoiseScale,
                    NoiseScaleW = _options.NoiseScaleW,
                    LengthScale = _options.LengthScale,
                };
                break;

            case SherpaOnnxTtsModelKind.Matcha:
                modelConfig.Matcha = new OfflineTtsMatchaModelConfig
                {
                    AcousticModel = _options.ModelPath,
                    Vocoder = ResolveVocoderPath(_options.ModelPath, _options.VocoderPath),
                    Tokens = _options.TokensPath,
                    Lexicon = _options.LexiconPath ?? ResolveSiblingFile(_options.ModelPath, "lexicon.txt") ?? string.Empty,
                    DataDir = _options.DataDirectory ?? ResolveSiblingDirectory(_options.ModelPath, "espeak-ng-data") ?? string.Empty,
                    DictDir = _options.DictDirectory ?? ResolveSiblingDirectory(_options.ModelPath, "dict") ?? string.Empty,
                    NoiseScale = _options.NoiseScale,
                    LengthScale = _options.LengthScale,
                };
                break;

            default:
                throw new NotSupportedException($"Unsupported SherpaOnnx TTS model kind: {_options.ModelKind}.");
        }

        return new OfflineTts(new OfflineTtsConfig
        {
            MaxNumSentences = 1,
            RuleFars = _options.RuleFars ?? string.Empty,
            RuleFsts = ResolveRuleFsts(_options.ModelPath, _options.RuleFsts),
            Model = modelConfig,
        });
    }

    private static string ResolveVocoderPath(string modelPath, string? vocoderPath)
    {
        if (!string.IsNullOrWhiteSpace(vocoderPath))
        {
            return vocoderPath;
        }

        var modelDirectory = Path.GetDirectoryName(modelPath);
        if (!string.IsNullOrWhiteSpace(modelDirectory))
        {
            var candidate = Path.Combine(modelDirectory, "vocos-16khz-univ.onnx");
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new InvalidOperationException(
            "SherpaOnnx Matcha TTS requires --tts-vocoder or a vocos-16khz-univ.onnx file next to --tts-model.");
    }

    private static string ResolveRuleFsts(string modelPath, string? ruleFsts)
    {
        if (!string.IsNullOrWhiteSpace(ruleFsts))
        {
            return ruleFsts;
        }

        var modelDirectory = Path.GetDirectoryName(modelPath);
        if (string.IsNullOrWhiteSpace(modelDirectory))
        {
            return string.Empty;
        }

        string[] preferredNames =
        [
            "phone-zh.fst",
            "date-zh.fst",
            "number-zh.fst",
        ];

        var resolved = preferredNames
            .Select(name => Path.Combine(modelDirectory, name))
            .Where(File.Exists)
            .ToArray();

        return resolved.Length > 0 ? string.Join(",", resolved) : string.Empty;
    }

    private static string? ResolveSiblingFile(string modelPath, string name)
    {
        var modelDirectory = Path.GetDirectoryName(modelPath);
        if (string.IsNullOrWhiteSpace(modelDirectory))
        {
            return null;
        }

        var candidate = Path.Combine(modelDirectory, name);
        return File.Exists(candidate) ? candidate : null;
    }

    private static string? ResolveSiblingDirectory(string modelPath, string name)
    {
        var modelDirectory = Path.GetDirectoryName(modelPath);
        if (string.IsNullOrWhiteSpace(modelDirectory))
        {
            return null;
        }

        var candidate = Path.Combine(modelDirectory, name);
        return Directory.Exists(candidate) ? candidate : null;
    }

    private static void ValidateFile(string path, string optionName)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"{optionName} points to a file that does not exist.", path);
        }
    }

    private static void ValidateOptionalFile(string? path, string optionName)
    {
        if (!string.IsNullOrWhiteSpace(path))
        {
            ValidateFile(path, optionName);
        }
    }
}
