using System.Runtime.InteropServices;
using Microsoft.Extensions.DependencyInjection;
using Zhengyan.QwenSharp.Models;
using Zhengyan.QwenSharp.Omni.Audio;
using Zhengyan.QwenSharp.OpenAI;
using Zhengyan.QwenSharp.OpenAI.Realtime;

var builder = WebApplication.CreateBuilder(args);
var serverOptions = builder.Configuration.GetSection("QwenSharp").Get<QwenSharpServerOptions>() ?? new QwenSharpServerOptions();
ApplyCommandLineOverrides(serverOptions, args);

var repositoryRoot = FindRepositoryRoot(builder.Environment.ContentRootPath);
var modelPath = ResolveModelPath(serverOptions, builder.Configuration);
var detectedModelType = QwenModelLoader.ReadModelType(modelPath);
var isOmni = serverOptions.ForceOmni || string.Equals(detectedModelType, "qwen2_5_omni", StringComparison.OrdinalIgnoreCase);
var modelName = string.IsNullOrWhiteSpace(serverOptions.ModelName) ? detectedModelType : serverOptions.ModelName;

if (serverOptions.NoCudaCache || GetBoolEnv("QWENSHARP_NO_CUDA_CACHE"))
{
    Environment.SetEnvironmentVariable("PYTORCH_NO_CUDA_MEMORY_CACHING", "1");
}

CopyModelsToOutputDirectory(repositoryRoot);

try
{
    if (isOmni)
    {
        var audioSynthesizer = CreateAudioSynthesizer(serverOptions.Tts, repositoryRoot, builder.Environment.ContentRootPath);
        builder.Services.AddSingleton(new Qwen25OmniOpenAIService(
            modelPath,
            serverOptions.DisableTalker,
            serverOptions.Device,
            serverOptions.DType,
            serverOptions.DeviceMap,
            audioSynthesizer));
        builder.Services.AddSingleton<IOpenAIChatCompletionsService>(sp => sp.GetRequiredService<Qwen25OmniOpenAIService>());
        builder.Services.AddSingleton<IOpenAIResponsesService>(sp => sp.GetRequiredService<Qwen25OmniOpenAIService>());
        builder.Services.AddSingleton<IOpenAIRealtimeSessionFactory, Qwen25OmniRealtimeSessionFactory>();
    }
    else
    {
        builder.Services.AddSingleton(new QwenTextOpenAIService(
            modelPath,
            serverOptions.Device,
            serverOptions.DType,
            modelName));
        builder.Services.AddSingleton<IOpenAIChatCompletionsService>(sp => sp.GetRequiredService<QwenTextOpenAIService>());
        builder.Services.AddSingleton<IOpenAIResponsesService>(sp => sp.GetRequiredService<QwenTextOpenAIService>());
    }
}
catch (DllNotFoundException ex)
{
    throw new InvalidOperationException(
        "Failed to initialize TorchSharp native libraries. On Linux, libLibTorchSharp.so is not enough by itself; libtorch.so must also be available to the dynamic loader. " +
        "Install the matching LibTorch/PyTorch native runtime and ensure its directory is included in LD_LIBRARY_PATH before starting the server.",
        ex);
}
catch (ExternalException ex) when (IsCudaOutOfMemory(ex) && !string.Equals(serverOptions.Device, "cpu", StringComparison.OrdinalIgnoreCase))
{
    throw new InvalidOperationException(
        $"CUDA out of memory while loading {modelName}. Try a smaller model, set QwenSharp:Device=cpu, set QwenSharp:DType=float16, set QwenSharp:DeviceMap, or set QwenSharp:DisableTalker=true for Omni models.",
        ex);
}

var app = builder.Build();
app.UseWebSockets();

app.MapGet("/", () => Results.Json(new
{
    name = "Zhengyan.QwenSharp.Server",
    model = modelName,
    model_type = detectedModelType,
    realtime = isOmni,
    endpoints = new[] { "/v1/chat/completions", "/v1/responses", isOmni ? "/v1/realtime" : null }.Where(endpoint => endpoint is not null),
}));

app.MapGet("/v1/models", () => Results.Json(new
{
    @object = "list",
    data = new[]
    {
        new
        {
            id = modelName,
            @object = "model",
            created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            owned_by = "qwen-sharp",
        },
    },
}));

app.MapOpenAICompatibleApi(new OpenAIEndpointOptions
{
    EnableRealtime = isOmni,
    DefaultRealtimeSession = new OpenAIRealtimeSessionOptions
    {
        Model = modelName,
        SampleRate = serverOptions.Realtime.SampleRate,
        ReturnAudio = serverOptions.Realtime.ReturnAudio,
        AutoRespond = serverOptions.Realtime.AutoRespond,
        UseAudioInVideo = serverOptions.Realtime.UseAudioInVideo,
        Voice = serverOptions.Realtime.Voice,
    },
});

app.Run();

static string ResolveModelPath(QwenSharpServerOptions options, IConfiguration configuration)
{
    if (!string.IsNullOrWhiteSpace(options.ModelPath))
    {
        return options.ModelPath;
    }

    var legacy = configuration["ModelPath"];
    if (!string.IsNullOrWhiteSpace(legacy))
    {
        return legacy;
    }

    throw new InvalidOperationException("Set QwenSharp:ModelPath in appsettings.json or pass --model-path <path>.");
}

static void ApplyCommandLineOverrides(QwenSharpServerOptions options, string[] args)
{
    SetStringArg(args, "--model-path", value => options.ModelPath = value);
    SetStringArg(args, "--model-name", value => options.ModelName = value);
    SetStringArg(args, "--device", value => options.Device = value);
    SetStringArg(args, "--dtype", value => options.DType = value);
    SetStringArg(args, "--device-map", value => options.DeviceMap = value);
    SetBoolArg(args, "--force-omni", value => options.ForceOmni = value);
    SetBoolArg(args, "--disable-talker", value => options.DisableTalker = value);
    SetBoolArg(args, "--no-cuda-cache", value => options.NoCudaCache = value);

    SetStringArg(args, "--tts-model", value => options.Tts.ModelPath = value);
    SetStringArg(args, "--tts-tokens", value => options.Tts.TokensPath = value);
    SetStringArg(args, "--tts-kind", value => options.Tts.Kind = value);
    SetStringArg(args, "--tts-provider", value => options.Tts.Provider = value);
    SetStringArg(args, "--tts-vocoder", value => options.Tts.VocoderPath = value);
    SetStringArg(args, "--tts-lexicon", value => options.Tts.LexiconPath = value);
    SetStringArg(args, "--tts-data-dir", value => options.Tts.DataDirectory = value);
    SetStringArg(args, "--tts-dict-dir", value => options.Tts.DictDirectory = value);
    SetStringArg(args, "--tts-rule-fars", value => options.Tts.RuleFars = value);
    SetStringArg(args, "--tts-rule-fsts", value => options.Tts.RuleFsts = value);
    SetFloatArg(args, "--tts-speed", value => options.Tts.Speed = value);
    SetIntArg(args, "--tts-speaker", value => options.Tts.SpeakerId = value);
}

static IQwen25OmniAudioSynthesizer? CreateAudioSynthesizer(
    QwenSharpTtsOptions options,
    string? repositoryRoot,
    string contentRoot)
{
    if (string.IsNullOrWhiteSpace(options.ModelPath) && string.IsNullOrWhiteSpace(options.TokensPath))
    {
        return null;
    }

    if (string.IsNullOrWhiteSpace(options.ModelPath) || string.IsNullOrWhiteSpace(options.TokensPath))
    {
        throw new InvalidOperationException("SherpaOnnx TTS requires both QwenSharp:Tts:ModelPath and QwenSharp:Tts:TokensPath.");
    }

    var kind = string.Equals(options.Kind, "matcha", StringComparison.OrdinalIgnoreCase)
        ? SherpaOnnxTtsModelKind.Matcha
        : SherpaOnnxTtsModelKind.Vits;

    return new SherpaOnnxAudioSynthesizer(new SherpaOnnxAudioSynthesizerOptions
    {
        ModelPath = ResolveExistingFile(options.ModelPath, "QwenSharp:Tts:ModelPath", repositoryRoot, contentRoot),
        TokensPath = ResolveExistingFile(options.TokensPath, "QwenSharp:Tts:TokensPath", repositoryRoot, contentRoot),
        ModelKind = kind,
        Provider = string.IsNullOrWhiteSpace(options.Provider) ? "cpu" : options.Provider,
        VocoderPath = ResolveOptionalFile(options.VocoderPath, "QwenSharp:Tts:VocoderPath", repositoryRoot, contentRoot),
        LexiconPath = ResolveOptionalFile(options.LexiconPath, "QwenSharp:Tts:LexiconPath", repositoryRoot, contentRoot),
        DataDirectory = ResolveOptionalDirectory(options.DataDirectory, "QwenSharp:Tts:DataDirectory", repositoryRoot, contentRoot),
        DictDirectory = ResolveOptionalDirectory(options.DictDirectory, "QwenSharp:Tts:DictDirectory", repositoryRoot, contentRoot),
        RuleFars = options.RuleFars,
        RuleFsts = options.RuleFsts,
        Speed = options.Speed,
        SpeakerId = options.SpeakerId,
    });
}

static string? FindRepositoryRoot(string contentRoot)
{
    var current = new DirectoryInfo(contentRoot);
    while (current is not null)
    {
        if (Directory.Exists(Path.Combine(current.FullName, "src"))
            && Directory.Exists(Path.Combine(current.FullName, "samples")))
        {
            return current.FullName;
        }

        current = current.Parent;
    }

    return null;
}

static void CopyModelsToOutputDirectory(string? repositoryRoot)
{
    if (string.IsNullOrWhiteSpace(repositoryRoot))
    {
        return;
    }

    var source = Path.Combine(repositoryRoot, "models");
    if (!Directory.Exists(source))
    {
        return;
    }

    CopyDirectoryPreserveNewest(source, Path.Combine(AppContext.BaseDirectory, "models"));
}

static void CopyDirectoryPreserveNewest(string sourceDirectory, string destinationDirectory)
{
    foreach (var directory in Directory.EnumerateDirectories(sourceDirectory, "*", SearchOption.AllDirectories))
    {
        var relative = Path.GetRelativePath(sourceDirectory, directory);
        Directory.CreateDirectory(Path.Combine(destinationDirectory, relative));
    }

    foreach (var file in Directory.EnumerateFiles(sourceDirectory, "*", SearchOption.AllDirectories))
    {
        var relative = Path.GetRelativePath(sourceDirectory, file);
        var destination = Path.Combine(destinationDirectory, relative);
        var destinationParent = Path.GetDirectoryName(destination);
        if (!string.IsNullOrWhiteSpace(destinationParent))
        {
            Directory.CreateDirectory(destinationParent);
        }

        var sourceInfo = new FileInfo(file);
        var destinationInfo = new FileInfo(destination);
        if (destinationInfo.Exists
            && destinationInfo.Length == sourceInfo.Length
            && destinationInfo.LastWriteTimeUtc >= sourceInfo.LastWriteTimeUtc)
        {
            continue;
        }

        File.Copy(file, destination, overwrite: true);
        File.SetLastWriteTimeUtc(destination, sourceInfo.LastWriteTimeUtc);
    }
}

static string ResolveExistingFile(string path, string optionName, string? repositoryRoot, string contentRoot)
{
    var resolved = ResolveExistingPath(path, optionName, repositoryRoot, contentRoot, Directory.Exists, File.Exists);
    if (!File.Exists(resolved))
    {
        throw new FileNotFoundException($"{optionName} points to a file that does not exist. Resolved path: {resolved}", resolved);
    }

    return resolved;
}

static string? ResolveOptionalFile(string? path, string optionName, string? repositoryRoot, string contentRoot)
    => string.IsNullOrWhiteSpace(path)
        ? null
        : ResolveExistingFile(path, optionName, repositoryRoot, contentRoot);

static string? ResolveOptionalDirectory(string? path, string optionName, string? repositoryRoot, string contentRoot)
{
    if (string.IsNullOrWhiteSpace(path))
    {
        return null;
    }

    var resolved = ResolveExistingPath(path, optionName, repositoryRoot, contentRoot, File.Exists, Directory.Exists);
    if (!Directory.Exists(resolved))
    {
        throw new DirectoryNotFoundException($"{optionName} points to a directory that does not exist. Resolved path: {resolved}");
    }

    return resolved;
}

static string ResolveExistingPath(
    string path,
    string optionName,
    string? repositoryRoot,
    string contentRoot,
    Func<string, bool> reject,
    Func<string, bool> accept)
{
    var expanded = Environment.ExpandEnvironmentVariables(ExpandHome(path));
    var candidates = new List<string>();

    if (Path.IsPathRooted(expanded))
    {
        candidates.Add(Path.GetFullPath(expanded));
    }
    else
    {
        candidates.Add(Path.GetFullPath(expanded, Environment.CurrentDirectory));
        candidates.Add(Path.GetFullPath(expanded, AppContext.BaseDirectory));
        candidates.Add(Path.GetFullPath(expanded, contentRoot));
        if (!string.IsNullOrWhiteSpace(repositoryRoot))
        {
            candidates.Add(Path.GetFullPath(expanded, repositoryRoot));
        }
    }

    foreach (var candidate in candidates.Distinct(StringComparer.OrdinalIgnoreCase))
    {
        if (accept(candidate))
        {
            return candidate;
        }

        if (reject(candidate))
        {
            break;
        }
    }

    throw new FileNotFoundException(
        $"{optionName} points to a path that does not exist. Tried: {string.Join(", ", candidates.Distinct(StringComparer.OrdinalIgnoreCase))}",
        path);
}

static string ExpandHome(string path)
{
    if (path == "~")
    {
        return Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    }

    return path.StartsWith("~/", StringComparison.Ordinal) || path.StartsWith("~\\", StringComparison.Ordinal)
        ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), path[2..])
        : path;
}

static bool GetBoolEnv(string name)
{
    var value = Environment.GetEnvironmentVariable(name);
    return string.Equals(value, "1", StringComparison.OrdinalIgnoreCase)
           || string.Equals(value, "true", StringComparison.OrdinalIgnoreCase)
           || string.Equals(value, "yes", StringComparison.OrdinalIgnoreCase);
}

static void SetStringArg(string[] args, string flag, Action<string> set)
{
    var value = GetArgValue(args, flag);
    if (!string.IsNullOrWhiteSpace(value))
    {
        set(value);
    }
}

static void SetBoolArg(string[] args, string flag, Action<bool> set)
{
    if (args.Any(arg => string.Equals(arg, flag, StringComparison.OrdinalIgnoreCase)))
    {
        set(true);
    }
}

static void SetIntArg(string[] args, string flag, Action<int> set)
{
    var value = GetArgValue(args, flag);
    if (int.TryParse(value, out var parsed))
    {
        set(parsed);
    }
}

static void SetFloatArg(string[] args, string flag, Action<float> set)
{
    var value = GetArgValue(args, flag);
    if (float.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var parsed))
    {
        set(parsed);
    }
}

static string? GetArgValue(string[] args, string flag)
{
    for (var i = 0; i < args.Length - 1; i++)
    {
        if (string.Equals(args[i], flag, StringComparison.OrdinalIgnoreCase))
        {
            return args[i + 1];
        }
    }

    return null;
}

static bool IsCudaOutOfMemory(Exception ex)
    => ex.Message.Contains("CUDA out of memory", StringComparison.OrdinalIgnoreCase)
       || ex.Message.Contains("CUDACachingAllocator", StringComparison.OrdinalIgnoreCase);

internal sealed class QwenSharpServerOptions
{
    public string? ModelPath { get; set; }

    public string? ModelName { get; set; }

    public bool ForceOmni { get; set; }

    public bool DisableTalker { get; set; }

    public bool NoCudaCache { get; set; }

    public string? Device { get; set; }

    public string? DType { get; set; }

    public string? DeviceMap { get; set; }

    public QwenSharpTtsOptions Tts { get; set; } = new();

    public QwenSharpRealtimeOptions Realtime { get; set; } = new();
}

internal sealed class QwenSharpTtsOptions
{
    public string? ModelPath { get; set; }

    public string? TokensPath { get; set; }

    public string Kind { get; set; } = "matcha";

    public string Provider { get; set; } = "cpu";

    public string? VocoderPath { get; set; }

    public string? LexiconPath { get; set; }

    public string? DataDirectory { get; set; }

    public string? DictDirectory { get; set; }

    public string? RuleFars { get; set; }

    public string? RuleFsts { get; set; }

    public float Speed { get; set; } = 1.0f;

    public int SpeakerId { get; set; }
}

internal sealed class QwenSharpRealtimeOptions
{
    public int SampleRate { get; set; } = 16_000;

    public bool ReturnAudio { get; set; } = true;

    public bool AutoRespond { get; set; } = true;

    public bool UseAudioInVideo { get; set; } = true;

    public string? Voice { get; set; }
}
