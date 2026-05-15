using System.Diagnostics;
using System.Globalization;
using System.Net.Http;
using System.Text.Json;
using Zhengyan.QwenSharp.Omni.Audio;
using Zhengyan.QwenSharp.OpenAI;
namespace Zhengyan.QwenSharp.OpenAI.Audio;

internal sealed record Qwen25OmniAudioAttachment(string Path, string Format, int TokenCount);

internal sealed record Qwen25OmniImageAttachment(
    string? Path,
    string? Url,
    string? FileId,
    string? Format,
    Qwen25OmniVisionTensor? VisionInput);

internal sealed record Qwen25OmniVideoAttachment(
    string? Path,
    string? Url,
    string? FileId,
    string? Format,
    bool IsTemporary,
    IReadOnlyList<Qwen25OmniVisionTensor>? FrameInputs);

internal sealed record Qwen25OmniMmInfo(
    IReadOnlyList<OpenAIMessage> Messages,
    IReadOnlyList<Qwen25OmniAudioAttachment> Audios,
    IReadOnlyList<Qwen25OmniImageAttachment> Images,
    IReadOnlyList<Qwen25OmniVideoAttachment> Videos,
    IReadOnlyList<OpenAIFileReference> Files,
    IReadOnlyList<Qwen25OmniVisionTensor> VisionInputs,
    string SystemPrompt,
    bool UseAudioInVideo,
    IReadOnlyList<string> Warnings,
    IReadOnlyList<string> TemporaryPaths) : IDisposable
{
    public bool HasUnsupportedMultimodalInputs => Images.Count > 0 || Videos.Count > 0;

    public void Dispose()
    {
        foreach (var input in VisionInputs)
        {
            input.Dispose();
        }

        foreach (var path in TemporaryPaths)
        {
            TryDelete(path);
        }
    }

    internal static void TryDelete(string path)
    {
        try
        {
            if (!string.IsNullOrWhiteSpace(path) && File.Exists(path))
            {
                File.Delete(path);
                return;
            }

            if (!string.IsNullOrWhiteSpace(path) && Directory.Exists(path))
            {
                Directory.Delete(path, recursive: true);
            }
        }
        catch
        {
        }
    }
}

internal static class Qwen25OmniMmInfoProcessor
{
    private static readonly HttpClient Http = new();
    private const int ImageFactor = 28;
    private const int FrameFactor = 2;
    private const int DefaultImageMinPixels = 4 * ImageFactor * ImageFactor;
    private const int DefaultImageMaxTokens = 1024;
    private const int DefaultVideoMinFrames = 4;
    private const int DefaultVideoMaxFrames = 8;
    private const int DefaultVideoMinPixels = 128 * ImageFactor * ImageFactor;
    private const int DefaultVideoMaxTokens = 768;
    private const int DefaultPositionIdPerSeconds = 25;

    private sealed record VideoProbeInfo(double? DurationSeconds, double? FrameRate, int? TotalFrames);

    private sealed record VideoFramePlan(int FrameCount, double SampleFps, int TemporalPositionStride);

    public static Qwen25OmniMmInfo ProcessMmInfo(
        IReadOnlyList<OpenAIMessage> messages,
        string systemPrompt,
        bool useAudioInVideo)
        => ProcessMmInfoAsync(messages, systemPrompt, useAudioInVideo).GetAwaiter().GetResult();

    public static async Task<Qwen25OmniMmInfo> ProcessMmInfoAsync(
        IReadOnlyList<OpenAIMessage> messages,
        string systemPrompt,
        bool useAudioInVideo,
        CancellationToken cancellationToken = default)
    {
        var normalizedMessages = messages.ToList();
        var audios = new List<Qwen25OmniAudioAttachment>();
        var images = new List<Qwen25OmniImageAttachment>();
        var videos = new List<Qwen25OmniVideoAttachment>();
        var visionInputs = new List<Qwen25OmniVisionTensor>();
        var files = new List<OpenAIFileReference>();
        var warnings = new List<string>();
        var temporaryPaths = new List<string>();
        var extractedVideoAudios = 0;

        foreach (var message in messages)
        {
            if (message.Parts is null)
            {
                continue;
            }

            foreach (var part in message.Parts)
            {
                if (string.Equals(part.Type, "input_audio", StringComparison.OrdinalIgnoreCase) && part.Audio is not null)
                {
                    var bytes = Convert.FromBase64String(part.Audio.DataBase64);
                    var tempPath = Path.Combine(Path.GetTempPath(), $"qwen_openai_{Guid.NewGuid():N}.{part.Audio.Format}");
                    File.WriteAllBytes(tempPath, bytes);
                    audios.Add(new Qwen25OmniAudioAttachment(tempPath, part.Audio.Format, EstimateAudioTokenCount(tempPath)));
                    temporaryPaths.Add(tempPath);
                }
                else if (string.Equals(part.Type, "image_url", StringComparison.OrdinalIgnoreCase) && part.ImageUrl is not null)
                {
                    var attachment = await ExtractImageAttachmentAsync(part.ImageUrl, temporaryPaths, cancellationToken).ConfigureAwait(false);
                    if (!string.IsNullOrWhiteSpace(attachment.Path))
                    {
                        var visionInput = BuildVisionInput(attachment.Path, tokenIndex: 151655);
                        visionInputs.Add(visionInput);
                        attachment = attachment with { VisionInput = visionInput };
                    }

                    images.Add(attachment);
                }
                else if (string.Equals(part.Type, "video", StringComparison.OrdinalIgnoreCase) && part.Video is not null)
                {
                    var videoAttachment = await ExtractVideoAttachmentAsync(part.Video, temporaryPaths, cancellationToken).ConfigureAwait(false);
                    if (!string.IsNullOrWhiteSpace(videoAttachment.Path) || !string.IsNullOrWhiteSpace(videoAttachment.Url) || !string.IsNullOrWhiteSpace(videoAttachment.FileId))
                    {
                        videos.Add(videoAttachment);
                        if (videoAttachment.FrameInputs is not null)
                        {
                            visionInputs.AddRange(videoAttachment.FrameInputs);
                        }
                        if (!string.IsNullOrWhiteSpace(videoAttachment.Path))
                        {
                            if (useAudioInVideo)
                            {
                                var audioAttachment = await TryExtractVideoAudioAsync(videoAttachment.Path, temporaryPaths, cancellationToken).ConfigureAwait(false);
                                if (audioAttachment is not null)
                                {
                                    audios.Add(audioAttachment);
                                    extractedVideoAudios++;
                                }
                            }
                        }
                    }
                }
                else if (string.Equals(part.Type, "file", StringComparison.OrdinalIgnoreCase) && part.File is not null)
                {
                    files.Add(part.File);
                }
            }
        }

        if (images.Count > 0)
        {
            var imageTokenCount = images.Sum(image => image.VisionInput?.TokenCount ?? 0);
            warnings.Add(
                $"Detected {images.Count} image input(s). Prepared {imageTokenCount} visual token(s) for the Omni vision stack.");
        }

        if (videos.Count > 0)
        {
            var videoTokenCount = videos.Sum(video => video.FrameInputs?.Sum(input => input.TokenCount) ?? 0);
            var videoVisionSummary = videoTokenCount > 0
                ? $" Prepared {videoTokenCount} video visual token(s) for the Omni vision stack."
                : " No video visual tokens were prepared.";

            if (useAudioInVideo && extractedVideoAudios > 0)
            {
                warnings.Add($"Detected {videos.Count} video input(s), extracted audio from {extractedVideoAudios} of them for use_audio_in_video=true.{videoVisionSummary}");
            }
            else if (useAudioInVideo)
            {
                warnings.Add($"Detected {videos.Count} video input(s). The adapter resolved the video paths, but could not extract audio tracks for use_audio_in_video=true.{videoVisionSummary}");
            }
            else
            {
                warnings.Add($"Detected {videos.Count} video input(s). The adapter resolved the video paths.{videoVisionSummary}");
            }
        }

        return new Qwen25OmniMmInfo(normalizedMessages, audios, images, videos, files, visionInputs, systemPrompt, useAudioInVideo, warnings, temporaryPaths);
    }

    private static async Task<Qwen25OmniImageAttachment> ExtractImageAttachmentAsync(
        OpenAIImageUrl image,
        ICollection<string> temporaryPaths,
        CancellationToken cancellationToken)
    {
        if (Uri.TryCreate(image.Url, UriKind.Absolute, out var uri))
        {
            if (string.Equals(uri.Scheme, Uri.UriSchemeFile, StringComparison.OrdinalIgnoreCase))
            {
                return new Qwen25OmniImageAttachment(Path.GetFullPath(uri.LocalPath), image.Url, null, null, null);
            }

            if (string.Equals(uri.Scheme, Uri.UriSchemeHttp, StringComparison.OrdinalIgnoreCase)
                || string.Equals(uri.Scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase))
            {
                var path = Path.Combine(Path.GetTempPath(), $"qwen_openai_{Guid.NewGuid():N}{InferExtension(uri, "png")}");
                var bytes = await Http.GetByteArrayAsync(uri, cancellationToken).ConfigureAwait(false);
                await File.WriteAllBytesAsync(path, bytes, cancellationToken).ConfigureAwait(false);
                temporaryPaths.Add(path);
                return new Qwen25OmniImageAttachment(path, image.Url, null, null, null);
            }
        }

        if (File.Exists(image.Url))
        {
            return new Qwen25OmniImageAttachment(Path.GetFullPath(image.Url), image.Url, null, null, null);
        }

        return new Qwen25OmniImageAttachment(null, image.Url, null, null, null);
    }

    private static Qwen25OmniVisionTensor BuildVisionInput(string path)
    {
        return BuildVisionInput(path, tokenIndex: 151655);
    }

    private static Qwen25OmniVisionTensor BuildVisionInput(string path, int tokenIndex)
    {
        return Qwen25OmniVisionProcessor.ProcessImage(
            path,
            patchSize: 14,
            mergeSize: 2,
            temporalPatchSize: 2,
            minPixels: DefaultImageMinPixels,
            maxPixels: GetImageMaxPixels(),
            tokenIndex);
    }

    private static int GetImageMaxPixels()
    {
        var tokenBudget = DefaultImageMaxTokens;
        var configured = Environment.GetEnvironmentVariable("QWENSHARP_OMNI_IMAGE_MAX_TOKENS");
        if (int.TryParse(configured, out var parsed))
        {
            tokenBudget = Math.Clamp(parsed, 64, 4096);
        }

        return tokenBudget * ImageFactor * ImageFactor;
    }

    private static int GetVideoMaxPixels()
    {
        var tokenBudget = DefaultVideoMaxTokens;
        var configured = Environment.GetEnvironmentVariable("QWENSHARP_OMNI_VIDEO_MAX_TOKENS");
        if (int.TryParse(configured, out var parsed))
        {
            tokenBudget = Math.Clamp(parsed, 128, 4096);
        }

        return tokenBudget * ImageFactor * ImageFactor;
    }

    private static int GetVideoMaxFrames()
    {
        var maxFrames = DefaultVideoMaxFrames;
        var configured = Environment.GetEnvironmentVariable("QWENSHARP_OMNI_VIDEO_MAX_FRAMES");
        if (int.TryParse(configured, out var parsed))
        {
            maxFrames = Math.Clamp(parsed, FrameFactor, 128);
        }

        return FloorToMultiple(maxFrames, FrameFactor);
    }

    private static async Task<Qwen25OmniAudioAttachment?> TryExtractVideoAudioAsync(
        string videoPath,
        ICollection<string> temporaryPaths,
        CancellationToken cancellationToken)
    {
        if (!File.Exists(videoPath))
        {
            return null;
        }

        var audioPath = Path.Combine(Path.GetTempPath(), $"qwen_openai_{Guid.NewGuid():N}.wav");

        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = "ffmpeg",
                UseShellExecute = false,
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                CreateNoWindow = true,
            };

            startInfo.ArgumentList.Add("-hide_banner");
            startInfo.ArgumentList.Add("-loglevel");
            startInfo.ArgumentList.Add("error");
            startInfo.ArgumentList.Add("-y");
            startInfo.ArgumentList.Add("-i");
            startInfo.ArgumentList.Add(videoPath);
            startInfo.ArgumentList.Add("-vn");
            startInfo.ArgumentList.Add("-ac");
            startInfo.ArgumentList.Add("1");
            startInfo.ArgumentList.Add("-ar");
            startInfo.ArgumentList.Add("16000");
            startInfo.ArgumentList.Add("-f");
            startInfo.ArgumentList.Add("wav");
            startInfo.ArgumentList.Add(audioPath);

            using var process = Process.Start(startInfo);
            if (process is null)
            {
                return null;
            }

            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
            if (process.ExitCode != 0 || !File.Exists(audioPath))
            {
                Qwen25OmniMmInfo.TryDelete(audioPath);
                return null;
            }

            temporaryPaths.Add(audioPath);
            return new Qwen25OmniAudioAttachment(audioPath, "wav", EstimateAudioTokenCount(audioPath));
        }
        catch
        {
            Qwen25OmniMmInfo.TryDelete(audioPath);
            return null;
        }
    }

    private static async Task<Qwen25OmniVideoAttachment> ExtractVideoAttachmentAsync(
        OpenAIVideoReference video,
        ICollection<string> temporaryPaths,
        CancellationToken cancellationToken)
    {
        var fileId = video.FileId;
        var format = string.IsNullOrWhiteSpace(video.Format) ? "mp4" : video.Format;

        if (!string.IsNullOrWhiteSpace(video.DataBase64))
        {
            var bytes = Convert.FromBase64String(video.DataBase64);
            var path = Path.Combine(Path.GetTempPath(), $"qwen_openai_{Guid.NewGuid():N}.{format}");
            File.WriteAllBytes(path, bytes);
            temporaryPaths.Add(path);
            var frameInputs = await TryExtractVideoFramesAsync(path, video, temporaryPaths, cancellationToken).ConfigureAwait(false);
            return new Qwen25OmniVideoAttachment(path, video.Url, fileId, format, true, frameInputs);
        }

        if (!string.IsNullOrWhiteSpace(video.Url))
        {
            if (Uri.TryCreate(video.Url, UriKind.Absolute, out var uri))
            {
                if (string.Equals(uri.Scheme, Uri.UriSchemeFile, StringComparison.OrdinalIgnoreCase))
                {
                    var path = uri.LocalPath;
                    var fullPath = Path.GetFullPath(path);
                    var frameInputs = await TryExtractVideoFramesAsync(fullPath, video, temporaryPaths, cancellationToken).ConfigureAwait(false);
                    return new Qwen25OmniVideoAttachment(fullPath, video.Url, fileId, format, false, frameInputs);
                }

                if (string.Equals(uri.Scheme, Uri.UriSchemeHttp, StringComparison.OrdinalIgnoreCase)
                    || string.Equals(uri.Scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase))
                {
                    var path = Path.Combine(Path.GetTempPath(), $"qwen_openai_{Guid.NewGuid():N}.{InferExtension(uri, format)}");
                    var bytes = await Http.GetByteArrayAsync(uri, cancellationToken).ConfigureAwait(false);
                    await File.WriteAllBytesAsync(path, bytes, cancellationToken).ConfigureAwait(false);
                    temporaryPaths.Add(path);
                    var frameInputs = await TryExtractVideoFramesAsync(path, video, temporaryPaths, cancellationToken).ConfigureAwait(false);
                    return new Qwen25OmniVideoAttachment(path, video.Url, fileId, format, true, frameInputs);
                }
            }

            if (File.Exists(video.Url))
            {
                var fullPath = Path.GetFullPath(video.Url);
                var frameInputs = await TryExtractVideoFramesAsync(fullPath, video, temporaryPaths, cancellationToken).ConfigureAwait(false);
                return new Qwen25OmniVideoAttachment(fullPath, video.Url, fileId, format, false, frameInputs);
            }
        }

        return new Qwen25OmniVideoAttachment(null, video.Url, fileId, format, false, null);
    }

    private static async Task<IReadOnlyList<Qwen25OmniVisionTensor>?> TryExtractVideoFramesAsync(
        string videoPath,
        OpenAIVideoReference video,
        ICollection<string> temporaryPaths,
        CancellationToken cancellationToken)
    {
        if (!File.Exists(videoPath))
        {
            return null;
        }

        var frameDir = Path.Combine(Path.GetTempPath(), $"qwen_openai_{Guid.NewGuid():N}_frames");
        Directory.CreateDirectory(frameDir);

        try
        {
            var probe = await ProbeVideoAsync(videoPath, cancellationToken).ConfigureAwait(false);
            var framePlan = ResolveVideoFramePlan(video, probe);
            var startInfo = new ProcessStartInfo
            {
                FileName = "ffmpeg",
                UseShellExecute = false,
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                CreateNoWindow = true,
            };

            startInfo.ArgumentList.Add("-hide_banner");
            startInfo.ArgumentList.Add("-loglevel");
            startInfo.ArgumentList.Add("error");
            startInfo.ArgumentList.Add("-y");
            startInfo.ArgumentList.Add("-i");
            startInfo.ArgumentList.Add(videoPath);

            if (video.VideoStart is not null)
            {
                startInfo.ArgumentList.Add("-ss");
                startInfo.ArgumentList.Add(video.VideoStart.Value.ToString(System.Globalization.CultureInfo.InvariantCulture));
            }

            if (video.VideoEnd is not null)
            {
                startInfo.ArgumentList.Add("-to");
                startInfo.ArgumentList.Add(video.VideoEnd.Value.ToString(System.Globalization.CultureInfo.InvariantCulture));
            }

            startInfo.ArgumentList.Add("-vf");
            startInfo.ArgumentList.Add($"fps={framePlan.SampleFps.ToString(CultureInfo.InvariantCulture)}");
            startInfo.ArgumentList.Add("-frames:v");
            startInfo.ArgumentList.Add(framePlan.FrameCount.ToString(CultureInfo.InvariantCulture));
            startInfo.ArgumentList.Add(Path.Combine(frameDir, "frame_%03d.png"));

            using var process = Process.Start(startInfo);
            if (process is null)
            {
                Qwen25OmniMmInfo.TryDelete(frameDir);
                return null;
            }

            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
            if (process.ExitCode != 0)
            {
                Qwen25OmniMmInfo.TryDelete(frameDir);
                return null;
            }

            var frameFiles = Directory.GetFiles(frameDir, "frame_*.png")
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            if (frameFiles.Length == 0)
            {
                Qwen25OmniMmInfo.TryDelete(frameDir);
                return null;
            }

            temporaryPaths.Add(frameDir);
            return
            [
                Qwen25OmniVisionProcessor.ProcessVideoFrames(
                    frameFiles,
                    patchSize: 14,
                    mergeSize: 2,
                    temporalPatchSize: 2,
                    minPixels: DefaultVideoMinPixels,
                    maxPixels: GetVideoMaxPixels(),
                    tokenIndex: 151656,
                    temporalPositionStride: framePlan.TemporalPositionStride)
            ];
        }
        catch
        {
            Qwen25OmniMmInfo.TryDelete(frameDir);
            return null;
        }
    }

    private static async Task<VideoProbeInfo> ProbeVideoAsync(string videoPath, CancellationToken cancellationToken)
    {
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = "ffprobe",
                UseShellExecute = false,
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                CreateNoWindow = true,
            };

            startInfo.ArgumentList.Add("-v");
            startInfo.ArgumentList.Add("error");
            startInfo.ArgumentList.Add("-select_streams");
            startInfo.ArgumentList.Add("v:0");
            startInfo.ArgumentList.Add("-show_entries");
            startInfo.ArgumentList.Add("stream=duration,avg_frame_rate,r_frame_rate,nb_frames:format=duration");
            startInfo.ArgumentList.Add("-of");
            startInfo.ArgumentList.Add("json");
            startInfo.ArgumentList.Add(videoPath);

            using var process = Process.Start(startInfo);
            if (process is null)
            {
                return new VideoProbeInfo(null, null, null);
            }

            var output = await process.StandardOutput.ReadToEndAsync(cancellationToken).ConfigureAwait(false);
            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
            if (process.ExitCode != 0 || string.IsNullOrWhiteSpace(output))
            {
                return new VideoProbeInfo(null, null, null);
            }

            using var document = JsonDocument.Parse(output);
            var root = document.RootElement;
            JsonElement? stream = null;
            if (root.TryGetProperty("streams", out var streams) &&
                streams.ValueKind == JsonValueKind.Array &&
                streams.GetArrayLength() > 0)
            {
                stream = streams[0];
            }

            var duration = stream is not null ? ReadDouble(stream.Value, "duration") : null;
            if (duration is null &&
                root.TryGetProperty("format", out var format) &&
                format.ValueKind == JsonValueKind.Object)
            {
                duration = ReadDouble(format, "duration");
            }

            var frameRate = stream is not null
                ? ParseRate(ReadString(stream.Value, "avg_frame_rate")) ?? ParseRate(ReadString(stream.Value, "r_frame_rate"))
                : null;

            var totalFrames = stream is not null ? ReadInt(stream.Value, "nb_frames") : null;
            if (totalFrames is null && duration is > 0 && frameRate is > 0)
            {
                totalFrames = Math.Max(FrameFactor, (int)Math.Round(duration.Value * frameRate.Value, MidpointRounding.AwayFromZero));
            }

            return new VideoProbeInfo(duration, frameRate, totalFrames);
        }
        catch
        {
            return new VideoProbeInfo(null, null, null);
        }
    }

    private static VideoFramePlan ResolveVideoFramePlan(OpenAIVideoReference video, VideoProbeInfo probe)
    {
        var sourceFps = probe.FrameRate is > 0 ? probe.FrameRate.Value : 2.0;
        var startSeconds = Math.Max(0.0, video.VideoStart ?? 0.0);
        var endSeconds = video.VideoEnd is > 0 ? video.VideoEnd.Value : probe.DurationSeconds;
        var clipDuration = endSeconds is > 0
            ? Math.Max(0.0, endSeconds.Value - startSeconds)
            : probe.DurationSeconds.GetValueOrDefault();

        if (clipDuration <= 0 && probe.TotalFrames is > 0)
        {
            clipDuration = probe.TotalFrames.Value / sourceFps;
        }

        var maxFrames = GetVideoMaxFrames();
        int frameCount;
        if (video.NFrames is > 0)
        {
            frameCount = RoundToNearestMultiple(video.NFrames.Value, FrameFactor);
            frameCount = Math.Clamp(frameCount, FrameFactor, maxFrames);
        }
        else
        {
            var requestedFps = video.Fps is > 0 ? video.Fps.Value : 2.0;
            var estimatedFrames = clipDuration > 0
                ? clipDuration * requestedFps
                : maxFrames;
            frameCount = FloorToMultiple(
                (int)Math.Clamp(estimatedFrames, DefaultVideoMinFrames, maxFrames),
                FrameFactor);
        }

        if (probe.TotalFrames is > 0)
        {
            var availableFrames = FloorToMultiple(probe.TotalFrames.Value, FrameFactor);
            if (availableFrames >= FrameFactor)
            {
                frameCount = Math.Min(frameCount, availableFrames);
            }
        }

        frameCount = Math.Max(FrameFactor, frameCount);
        var sampleFps = clipDuration > 0
            ? frameCount / clipDuration
            : video.Fps is > 0
                ? video.Fps.Value
                : Math.Min(sourceFps, frameCount);
        sampleFps = Math.Max(0.1, sampleFps);

        var temporalStride = Math.Max(
            1,
            (int)Math.Floor((2.0 / sampleFps) * DefaultPositionIdPerSeconds));

        return new VideoFramePlan(frameCount, sampleFps, temporalStride);
    }

    private static double? ReadDouble(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out var property))
        {
            return null;
        }

        return property.ValueKind switch
        {
            JsonValueKind.Number when property.TryGetDouble(out var value) => value,
            JsonValueKind.String when double.TryParse(property.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var value) => value,
            _ => null,
        };
    }

    private static int? ReadInt(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out var property))
        {
            return null;
        }

        return property.ValueKind switch
        {
            JsonValueKind.Number when property.TryGetInt32(out var value) => value,
            JsonValueKind.String when int.TryParse(property.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var value) => value,
            _ => null,
        };
    }

    private static string? ReadString(JsonElement element, string propertyName)
        => element.TryGetProperty(propertyName, out var property) && property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;

    private static double? ParseRate(string? value)
    {
        if (string.IsNullOrWhiteSpace(value) || string.Equals(value, "0/0", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        var parts = value.Split('/', 2);
        if (parts.Length == 2 &&
            double.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out var numerator) &&
            double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var denominator) &&
            denominator != 0)
        {
            return numerator / denominator;
        }

        return double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed)
            ? parsed
            : null;
    }

    private static int RoundToNearestMultiple(int value, int factor)
        => Math.Max(factor, (int)Math.Round((double)value / factor, MidpointRounding.AwayFromZero) * factor);

    private static int FloorToMultiple(int value, int factor)
        => Math.Max(factor, (int)Math.Floor((double)value / factor) * factor);

    private static int EstimateAudioTokenCount(string wavPath)
    {
        using var mel = Qwen25OmniAudioProcessor.LoadMelSpectrogram(wavPath);
        return Qwen25OmniAudioProcessor.EstimateAudioTokenCount(mel);
    }

    private static string InferExtension(Uri uri, string fallback)
    {
        var extension = Path.GetExtension(uri.AbsolutePath);
        if (!string.IsNullOrWhiteSpace(extension))
        {
            return extension.TrimStart('.');
        }

        return fallback;
    }
}
