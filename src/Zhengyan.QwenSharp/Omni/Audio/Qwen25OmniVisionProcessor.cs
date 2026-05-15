using System;
using System.Collections.Generic;
using System.IO;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace Zhengyan.QwenSharp.Omni.Audio;

public sealed record Qwen25OmniVisionTensor(
    Tensor PixelValues,
    Tensor ImageGridThw,
    int TokenIndex,
    int TokenCount,
    int TemporalPositionStride = 1) : IDisposable
{
    public void Dispose()
    {
        PixelValues.Dispose();
        ImageGridThw.Dispose();
    }
}

public static class Qwen25OmniVisionProcessor
{
    private static readonly io.SkiaImager SkiaImageIo = new(95);
    private static readonly float[] ImageMean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] ImageStd = [0.26862954f, 0.26130258f, 0.27577711f];

    public static Qwen25OmniVisionTensor ProcessImage(
        string imagePath,
        int patchSize,
        int mergeSize,
        int temporalPatchSize,
        int minPixels,
        int maxPixels,
        int tokenIndex)
    {
        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"Image not found: {imagePath}");
        }

        using var image = io.read_image(imagePath, io.ImageReadMode.RGB, SkiaImageIo);
        var sourceHeight = (int)image.shape[1];
        var sourceWidth = (int)image.shape[2];
        var factor = patchSize * mergeSize;
        var (targetHeight, targetWidth) = SmartResize(sourceHeight, sourceWidth, factor, minPixels, maxPixels);
        using var imageFloat = image.to_type(ScalarType.Float32) / 255.0f;
        using var resized = transforms.functional.resize(imageFloat, targetHeight, targetWidth);
        var pixels = resized.contiguous().data<float>().ToArray();

        int gridT = 1;
        int gridH = targetHeight / patchSize;
        int gridW = targetWidth / patchSize;
        int mergedGridH = gridH / mergeSize;
        int mergedGridW = gridW / mergeSize;
        int patchCount = gridT * gridH * gridW;
        int patchVectorSize = 3 * temporalPatchSize * patchSize * patchSize;
        var pixelBuffer = new float[patchCount * patchVectorSize];

        int index = 0;
        for (int t = 0; t < gridT; t++)
        {
            for (int mergedY = 0; mergedY < mergedGridH; mergedY++)
            {
                for (int mergedX = 0; mergedX < mergedGridW; mergedX++)
                {
                    for (int mergeInnerY = 0; mergeInnerY < mergeSize; mergeInnerY++)
                    {
                        for (int mergeInnerX = 0; mergeInnerX < mergeSize; mergeInnerX++)
                        {
                            for (int channel = 0; channel < 3; channel++)
                            {
                                for (int temporal = 0; temporal < temporalPatchSize; temporal++)
                                {
                                    for (int patchY = 0; patchY < patchSize; patchY++)
                                    {
                                        for (int patchX = 0; patchX < patchSize; patchX++)
                                        {
                                            int x = ((mergedX * mergeSize + mergeInnerX) * patchSize) + patchX;
                                            int y = ((mergedY * mergeSize + mergeInnerY) * patchSize) + patchY;
                                            int pixelIndex = channel * targetHeight * targetWidth + y * targetWidth + x;
                                            float value = pixels[pixelIndex];
                                            pixelBuffer[index++] = (value - ImageMean[channel]) / ImageStd[channel];
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
        var tokenCount = Math.Max(1, patchCount / (mergeSize * mergeSize));
        return new Qwen25OmniVisionTensor(pixelValues, imageGridThw, tokenIndex, tokenCount);
    }

    public static Qwen25OmniVisionTensor ProcessVideoFrames(
        IReadOnlyList<string> framePaths,
        int patchSize,
        int mergeSize,
        int temporalPatchSize,
        int minPixels,
        int maxPixels,
        int tokenIndex,
        int temporalPositionStride)
    {
        if (framePaths.Count == 0)
        {
            throw new ArgumentException("At least one video frame is required.", nameof(framePaths));
        }

        foreach (var framePath in framePaths)
        {
            if (!File.Exists(framePath))
            {
                throw new FileNotFoundException($"Video frame not found: {framePath}");
            }
        }

        using var firstImage = io.read_image(framePaths[0], io.ImageReadMode.RGB, SkiaImageIo);
        var sourceHeight = (int)firstImage.shape[1];
        var sourceWidth = (int)firstImage.shape[2];
        var factor = patchSize * mergeSize;
        var (targetHeight, targetWidth) = SmartResize(sourceHeight, sourceWidth, factor, minPixels, maxPixels);
        var paddedFrameCount = CeilToMultiple(framePaths.Count, temporalPatchSize);
        var framePixels = new float[paddedFrameCount][];

        for (var i = 0; i < framePaths.Count; i++)
        {
            framePixels[i] = ReadResizedPixels(framePaths[i], targetHeight, targetWidth);
        }

        for (var i = framePaths.Count; i < paddedFrameCount; i++)
        {
            framePixels[i] = framePixels[framePaths.Count - 1];
        }

        int gridT = paddedFrameCount / temporalPatchSize;
        int gridH = targetHeight / patchSize;
        int gridW = targetWidth / patchSize;
        int mergedGridH = gridH / mergeSize;
        int mergedGridW = gridW / mergeSize;
        int patchCount = gridT * gridH * gridW;
        int patchVectorSize = 3 * temporalPatchSize * patchSize * patchSize;
        var pixelBuffer = new float[patchCount * patchVectorSize];

        int index = 0;
        for (int t = 0; t < gridT; t++)
        {
            for (int mergedY = 0; mergedY < mergedGridH; mergedY++)
            {
                for (int mergedX = 0; mergedX < mergedGridW; mergedX++)
                {
                    for (int mergeInnerY = 0; mergeInnerY < mergeSize; mergeInnerY++)
                    {
                        for (int mergeInnerX = 0; mergeInnerX < mergeSize; mergeInnerX++)
                        {
                            for (int channel = 0; channel < 3; channel++)
                            {
                                for (int temporal = 0; temporal < temporalPatchSize; temporal++)
                                {
                                    var frameIndex = (t * temporalPatchSize) + temporal;
                                    var pixels = framePixels[frameIndex];
                                    for (int patchY = 0; patchY < patchSize; patchY++)
                                    {
                                        for (int patchX = 0; patchX < patchSize; patchX++)
                                        {
                                            int x = ((mergedX * mergeSize + mergeInnerX) * patchSize) + patchX;
                                            int y = ((mergedY * mergeSize + mergeInnerY) * patchSize) + patchY;
                                            int pixelIndex = channel * targetHeight * targetWidth + y * targetWidth + x;
                                            float value = pixels[pixelIndex];
                                            pixelBuffer[index++] = (value - ImageMean[channel]) / ImageStd[channel];
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
        var tokenCount = Math.Max(1, patchCount / (mergeSize * mergeSize));
        return new Qwen25OmniVisionTensor(pixelValues, imageGridThw, tokenIndex, tokenCount, temporalPositionStride);
    }

    private static float[] ReadResizedPixels(string imagePath, int targetHeight, int targetWidth)
    {
        using var image = io.read_image(imagePath, io.ImageReadMode.RGB, SkiaImageIo);
        using var imageFloat = image.to_type(ScalarType.Float32) / 255.0f;
        using var resized = transforms.functional.resize(imageFloat, targetHeight, targetWidth);
        return resized.contiguous().data<float>().ToArray();
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
        => Math.Max(factor, (int)Math.Round((double)value / factor, MidpointRounding.AwayFromZero) * factor);

    private static int CeilToMultiple(int value, int factor)
        => Math.Max(factor, (int)Math.Ceiling((double)value / factor) * factor);
}
