using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Zhengyan.QwenSharp.Tokenizers;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace Zhengyan.QwenSharp.Vision;

public sealed class QwenVisionInput : IDisposable
{
    public required int[] InputIds { get; init; }
    public required Tensor PixelValues { get; init; }
    public required Tensor ImageGridThw { get; init; }

    public void Dispose()
    {
        PixelValues.Dispose();
        ImageGridThw.Dispose();
    }
}

public sealed class QwenVisionProcessorConfig
{
    public required string ModelType { get; init; }
    public required int ImageTokenId { get; init; }
    public required int VisionStartTokenId { get; init; }
    public required int VisionEndTokenId { get; init; }
    public required int PatchSize { get; init; }
    public required int TemporalPatchSize { get; init; }
    public required int MergeSize { get; init; }
    public required int MinPixels { get; init; }
    public required int MaxPixels { get; init; }
    public required bool NeedsEmptyThinkBlock { get; init; }

    public static QwenVisionProcessorConfig FromDirectory(string modelDirectory)
    {
        using var configDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(modelDirectory, "config.json")));
        using var preprocessorDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(modelDirectory, "preprocessor_config.json")));

        var root = configDoc.RootElement;
        var preprocessor = preprocessorDoc.RootElement;
        var modelType = root.GetProperty("model_type").GetString() ?? string.Empty;
        var size = preprocessor.GetProperty("size");

        return new QwenVisionProcessorConfig
        {
            ModelType = modelType,
            ImageTokenId = root.GetProperty("image_token_id").GetInt32(),
            VisionStartTokenId = root.GetProperty("vision_start_token_id").GetInt32(),
            VisionEndTokenId = root.GetProperty("vision_end_token_id").GetInt32(),
            PatchSize = preprocessor.GetProperty("patch_size").GetInt32(),
            TemporalPatchSize = preprocessor.GetProperty("temporal_patch_size").GetInt32(),
            MergeSize = preprocessor.GetProperty("merge_size").GetInt32(),
            MinPixels = size.GetProperty("shortest_edge").GetInt32(),
            MaxPixels = size.GetProperty("longest_edge").GetInt32(),
            NeedsEmptyThinkBlock = string.Equals(modelType, "qwen3_5", StringComparison.OrdinalIgnoreCase),
        };
    }
}

public static class QwenVisionProcessor
{
    private static readonly io.SkiaImager SkiaImageIo = new(95);
    public static QwenVisionInput PrepareImagePrompt(
        string modelDirectory,
        Qwen2Tokenizer tokenizer,
        string imagePath,
        string instruction)
    {
        var config = QwenVisionProcessorConfig.FromDirectory(modelDirectory);
        var (pixelValues, imageGridThw, imageTokenCount) = ProcessImage(imagePath, config);
        var inputIds = BuildInputIds(tokenizer, config, instruction, imageTokenCount);

        return new QwenVisionInput
        {
            InputIds = inputIds,
            PixelValues = pixelValues,
            ImageGridThw = imageGridThw,
        };
    }

    private static int[] BuildInputIds(Qwen2Tokenizer tokenizer, QwenVisionProcessorConfig config, string instruction, int imageTokenCount)
    {
        var ids = new List<int>();
        ids.Add(tokenizer.ImStartId);
        ids.AddRange(tokenizer.Encode("user\n", addSpecialTokens: false));
        ids.Add(config.VisionStartTokenId);
        ids.AddRange(Enumerable.Repeat(config.ImageTokenId, imageTokenCount));
        ids.Add(config.VisionEndTokenId);
        ids.AddRange(tokenizer.Encode(instruction, addSpecialTokens: false));
        ids.Add(tokenizer.ImEndId);
        ids.AddRange(tokenizer.Encode("\n", addSpecialTokens: false));
        ids.Add(tokenizer.ImStartId);
        ids.AddRange(tokenizer.Encode("assistant\n", addSpecialTokens: false));

        if (config.NeedsEmptyThinkBlock)
        {
            ids.AddRange(tokenizer.Encode("<think>", addSpecialTokens: true));
            ids.AddRange(tokenizer.Encode("\n\n", addSpecialTokens: false));
            ids.AddRange(tokenizer.Encode("</think>", addSpecialTokens: true));
            ids.AddRange(tokenizer.Encode("\n\n", addSpecialTokens: false));
        }

        return ids.ToArray();
    }

    private static (Tensor pixelValues, Tensor imageGridThw, int imageTokenCount) ProcessImage(string imagePath, QwenVisionProcessorConfig config)
    {
        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"Image not found: {imagePath}");
        }

        using var image = io.read_image(imagePath, io.ImageReadMode.RGB, SkiaImageIo);
        var sourceHeight = (int)image.shape[1];
        var sourceWidth = (int)image.shape[2];
        var factor = config.PatchSize * config.MergeSize;
        var (targetHeight, targetWidth) = SmartResize(sourceHeight, sourceWidth, factor, config.MinPixels, config.MaxPixels);
        using var imageFloat = image.to_type(ScalarType.Float32) / 255.0f;
        using var resized = transforms.functional.resize(imageFloat, targetHeight, targetWidth);
        var pixels = resized.contiguous().data<float>().ToArray();

        int gridT = 1;
        int gridH = targetHeight / config.PatchSize;
        int gridW = targetWidth / config.PatchSize;
        int mergedGridH = gridH / config.MergeSize;
        int mergedGridW = gridW / config.MergeSize;
        int imageTokenCount = gridT * mergedGridH * mergedGridW;
        int patchCount = gridT * gridH * gridW;
        int patchVectorSize = 3 * config.TemporalPatchSize * config.PatchSize * config.PatchSize;
        var pixelBuffer = new float[patchCount * patchVectorSize];

        int index = 0;
        for (int t = 0; t < gridT; t++)
        {
            for (int mergedY = 0; mergedY < mergedGridH; mergedY++)
            {
                for (int mergedX = 0; mergedX < mergedGridW; mergedX++)
                {
                    for (int mergeInnerY = 0; mergeInnerY < config.MergeSize; mergeInnerY++)
                    {
                        for (int mergeInnerX = 0; mergeInnerX < config.MergeSize; mergeInnerX++)
                        {
                            for (int channel = 0; channel < 3; channel++)
                            {
                                for (int temporal = 0; temporal < config.TemporalPatchSize; temporal++)
                                {
                                    for (int patchY = 0; patchY < config.PatchSize; patchY++)
                                    {
                                        for (int patchX = 0; patchX < config.PatchSize; patchX++)
                                        {
                                            int x = ((mergedX * config.MergeSize + mergeInnerX) * config.PatchSize) + patchX;
                                            int y = ((mergedY * config.MergeSize + mergeInnerY) * config.PatchSize) + patchY;
                                            int pixelIndex = channel * targetHeight * targetWidth + y * targetWidth + x;
                                            float value = pixels[pixelIndex];
                                            pixelBuffer[index++] = (value - 0.5f) / 0.5f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        using var scope = NewDisposeScope();
        var pixelValuesBase = tensor(pixelBuffer, dtype: ScalarType.Float32);
        var imageGridThwBase = tensor(new long[] { gridT, gridH, gridW }, dtype: ScalarType.Int64);
        var pixelValues = scope.MoveToOuter(pixelValuesBase.view(patchCount, patchVectorSize));
        var imageGridThw = scope.MoveToOuter(imageGridThwBase.view(1, 3));
        return (pixelValues, imageGridThw, imageTokenCount);
    }

    private static (int height, int width) SmartResize(int height, int width, int factor, int minPixels, int maxPixels)
    {
        if ((double)Math.Max(height, width) / Math.Min(height, width) > 200.0)
        {
            throw new InvalidOperationException("Image aspect ratio must be smaller than 200.");
        }

        int resizedHeight = RoundToNearestMultiple(height, factor);
        int resizedWidth = RoundToNearestMultiple(width, factor);
        long pixelCount = (long)resizedHeight * resizedWidth;

        if (pixelCount > maxPixels)
        {
            double beta = Math.Sqrt((double)(height * width) / maxPixels);
            resizedHeight = Math.Max(factor, (int)Math.Floor(height / beta / factor) * factor);
            resizedWidth = Math.Max(factor, (int)Math.Floor(width / beta / factor) * factor);
        }
        else if (pixelCount < minPixels)
        {
            double beta = Math.Sqrt((double)minPixels / (height * width));
            resizedHeight = (int)Math.Ceiling(height * beta / factor) * factor;
            resizedWidth = (int)Math.Ceiling(width * beta / factor) * factor;
        }

        return (resizedHeight, resizedWidth);
    }

    private static int RoundToNearestMultiple(int value, int factor)
    {
        return Math.Max(factor, (int)Math.Round((double)value / factor, MidpointRounding.AwayFromZero) * factor);
    }
}

