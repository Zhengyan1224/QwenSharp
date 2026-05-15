using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using Zhengyan.QwenSharp.Generation;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.Core;

/// <summary>
/// Utility helpers for working with TorchSharp tensors.
/// </summary>
public static class TorchHelper
{
    /// <summary>
    /// Maps HuggingFace torch_dtype strings to TorchSharp ScalarType.
    /// </summary>
    public static ScalarType ParseDtype(string? torchDtype) => torchDtype switch
    {
        "float16" or "fp16" => ScalarType.Float16,
        "bfloat16" or "bf16" => ScalarType.BFloat16,
        "float32" or "fp32" or "float" => ScalarType.Float32,
        "float64" or "fp64" or "double" => ScalarType.Float64,
        "int8" => ScalarType.Int8,
        "int16" => ScalarType.Int16,
        "int32" or "int" => ScalarType.Int32,
        "int64" or "long" => ScalarType.Int64,
        _ => ScalarType.Float32, // default
    };

    /// <summary>
    /// Parses a device string like "cpu", "cuda", "cuda:0", "cuda:1".
    /// </summary>
    public static Device ParseDevice(string? deviceStr)
    {
        deviceStr = GetDeviceOverride(deviceStr);

        if (string.IsNullOrEmpty(deviceStr) || deviceStr == "cpu")
            return new Device(DeviceType.CPU);

        if (deviceStr.StartsWith("cuda", StringComparison.OrdinalIgnoreCase))
        {
            if (deviceStr.Contains(':'))
            {
                var parts = deviceStr.Split(':');
                if (int.TryParse(parts[1], out var index))
                    return new Device(DeviceType.CUDA, index);
            }
            return new Device(DeviceType.CUDA);
        }

        throw new ArgumentException($"Unsupported device: '{deviceStr}'. Use 'cpu' or 'cuda[:<index>]'.");
    }

    /// <summary>
    /// Resolves a device string from an explicit argument or environment variable.
    /// </summary>
    public static string? GetDeviceOverride(string? explicitDevice = null)
    {
        if (!string.IsNullOrWhiteSpace(explicitDevice))
        {
            return explicitDevice.Trim();
        }

        var envDevice = Environment.GetEnvironmentVariable("QWENSHARP_DEVICE");
        if (!string.IsNullOrWhiteSpace(envDevice))
        {
            return envDevice.Trim();
        }

        return null;
    }

    /// <summary>
    /// Parses a comma-separated multi-device map like "cuda:0,cuda:1".
    /// Use "auto" to expand to all visible CUDA devices in index order.
    /// </summary>
    public static IReadOnlyList<Device> ParseDeviceMap(string? deviceMapStr)
    {
        if (string.IsNullOrWhiteSpace(deviceMapStr))
        {
            return Array.Empty<Device>();
        }

        if (string.Equals(deviceMapStr.Trim(), "auto", StringComparison.OrdinalIgnoreCase))
        {
            if (!IsCudaAvailable())
            {
                return Array.Empty<Device>();
            }

            var count = torch.cuda.device_count();
            return Enumerable.Range(0, count)
                .Select(index => new Device(DeviceType.CUDA, index))
                .ToArray();
        }

        var devices = deviceMapStr
            .Split([',', ';', '|'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(ParseDevice)
            .ToArray();

        return devices;
    }

    /// <summary>
    /// Resolves a tensor/model dtype from an explicit argument or environment variable.
    /// </summary>
    public static ScalarType? ResolveModelDtype(string? explicitDtype = null, Device? device = null, string? configDtype = null)
    {
        var dtype = GetDtypeOverride(explicitDtype) ?? GetDtypeOverride(Environment.GetEnvironmentVariable("QWENSHARP_DTYPE"));

        if (!string.IsNullOrWhiteSpace(dtype))
        {
            return ParseDtype(dtype);
        }

        if (!string.IsNullOrWhiteSpace(configDtype))
        {
            var parsed = ParseDtype(configDtype);
            if (device is { type: DeviceType.CUDA } && parsed == ScalarType.BFloat16)
            {
                // V100 and similar cards do not have native BF16 support.
                return ScalarType.Float16;
            }

            return parsed;
        }

        return device is { type: DeviceType.CUDA } ? ScalarType.Float16 : ScalarType.Float32;
    }

    /// <summary>
    /// Returns an override dtype value if one was supplied.
    /// </summary>
    public static string? GetDtypeOverride(string? explicitDtype = null)
    {
        if (!string.IsNullOrWhiteSpace(explicitDtype))
        {
            return explicitDtype.Trim();
        }

        return null;
    }

    /// <summary>
    /// Initializes TorchSharp for the selected runtime device.
    /// </summary>
    public static void InitializeRuntime(Device device)
    {
        torch.InitializeDeviceType(device.type == DeviceType.CUDA ? DeviceType.CUDA : DeviceType.CPU);
    }

    /// <summary>
    /// Converts logits to a TorchSharp-safe dtype for sampling and scalar extraction.
    /// TorchSharp does not support item&lt;float&gt;() on BF16 tensors.
    /// </summary>
    public static Tensor PrepareSamplingLogits(Tensor logits)
        => logits.dtype == ScalarType.Float32 ? logits : logits.to(ScalarType.Float32);

    /// <summary>
    /// Samples a token id from logits using CPU-side top-k/top-p sampling.
    /// This avoids CUDA-side multinomial/device-assert failures on unstable half precision outputs.
    /// </summary>
    public static long SampleTokenId(Tensor logits, GenerationConfig config)
    {
        using var prepared = PrepareSamplingLogits(logits);
        using var cpuLogits = prepared.to(DeviceType.CPU);
        var scores = cpuLogits.data<float>().ToArray();

        if (!config.DoSample)
        {
            return ArgMax(scores);
        }

        if (config.Temperature > 0 && config.Temperature != 1.0f)
        {
            var temperature = config.Temperature;
            for (var i = 0; i < scores.Length; i++)
            {
                scores[i] /= temperature;
            }
        }

        for (var i = 0; i < scores.Length; i++)
        {
            if (!float.IsFinite(scores[i]))
            {
                scores[i] = float.NegativeInfinity;
            }
        }

        var selected = BuildCandidateSet(scores, config.TopK, config.TopP);
        if (selected.Count == 0)
        {
            return ArgMax(scores);
        }

        var maxLogit = selected.Max(item => item.Logit);
        var probabilities = new double[selected.Count];
        var total = 0.0;
        for (var i = 0; i < selected.Count; i++)
        {
            var probability = Math.Exp(selected[i].Logit - maxLogit);
            probabilities[i] = probability;
            total += probability;
        }

        if (!(total > 0) || double.IsNaN(total) || double.IsInfinity(total))
        {
            return ArgMax(scores);
        }

        var sample = Random.Shared.NextDouble() * total;
        var cumulative = 0.0;
        for (var i = 0; i < selected.Count; i++)
        {
            cumulative += probabilities[i];
            if (sample <= cumulative)
            {
                return selected[i].Index;
            }
        }

        return selected[^1].Index;
    }

    private static long ArgMax(IReadOnlyList<float> scores)
    {
        var bestIndex = 0;
        var bestScore = float.NegativeInfinity;
        for (var i = 0; i < scores.Count; i++)
        {
            if (scores[i] > bestScore)
            {
                bestScore = scores[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    private static List<(int Index, float Logit)> BuildCandidateSet(float[] scores, int topK, float topP)
    {
        var ranked = scores
            .Select((logit, index) => (Index: index, Logit: logit))
            .OrderByDescending(item => item.Logit)
            .ToList();

        if (topK > 0 && topK < ranked.Count)
        {
            ranked = ranked.Take(topK).ToList();
        }

        if (topP < 1.0f)
        {
            var maxLogit = ranked.Count > 0 ? ranked[0].Logit : float.NegativeInfinity;
            var expValues = new double[ranked.Count];
            var total = 0.0;
            for (var i = 0; i < ranked.Count; i++)
            {
                var value = Math.Exp(ranked[i].Logit - maxLogit);
                expValues[i] = value;
                total += value;
            }

            if (total > 0 && !double.IsNaN(total) && !double.IsInfinity(total))
            {
                var cumulative = 0.0;
                var filtered = new List<(int Index, float Logit)>();
                for (var i = 0; i < ranked.Count; i++)
                {
                    cumulative += expValues[i] / total;
                    filtered.Add(ranked[i]);
                    if (cumulative >= topP)
                    {
                        break;
                    }
                }

                if (filtered.Count > 0)
                {
                    return filtered;
                }
            }
        }

        return ranked;
    }

    /// <summary>
    /// Loads a named parameter tensor from a weight dictionary into a module parameter.
    /// Handles dtype casting and device placement.
    /// </summary>
    public static void LoadParameter(
        nn.Module module,
        string paramName,
        Dictionary<string, Tensor> weights,
        ScalarType targetDtype,
        Device targetDevice)
    {
        if (!weights.TryGetValue(paramName, out var srcTensor))
        {
            Console.Error.WriteLine($"[WARN] Weight '{paramName}' not found in checkpoint.");
            return;
        }

        var param = module.get_parameter(paramName.Split('.')[^1]);
        if (param is null) return;

        using var castTensor = srcTensor.to(targetDtype, targetDevice);
        param.copy_(castTensor);
    }

    /// <summary>
    /// Returns a human-readable string describing a tensor's shape and dtype.
    /// </summary>
    public static string DescribeTensor(Tensor t)
        => $"[{string.Join(", ", t.shape)}] {t.dtype} on {t.device}";

    /// <summary>
    /// Computes the product of all elements in an array (total element count).
    /// </summary>
    public static long ElementCount(long[] shape)
    {
        if (shape.Length == 0) return 1;
        var count = 1L;
        foreach (var d in shape) count *= d;
        return count;
    }

    /// <summary>
    /// Returns true if CUDA is available on this machine.
    /// </summary>
    public static bool IsCudaAvailable() => torch.cuda.is_available();

    /// <summary>
    /// Returns the best available device (CUDA if available, else CPU).
    /// </summary>
    public static Device GetDefaultDevice()
        => ParseDevice(Environment.GetEnvironmentVariable("QWENSHARP_DEVICE") ?? (IsCudaAvailable() ? "cuda" : "cpu"));

    /// <summary>
    /// Prints a brief diagnostic of the TorchSharp environment.
    /// </summary>
    public static void PrintEnvironmentInfo()
    {
        Console.WriteLine($"TorchSharp version: {typeof(torch).Assembly.GetName().Version}");
        Console.WriteLine($"CUDA available: {IsCudaAvailable()}");
        if (IsCudaAvailable())
        {
            Console.WriteLine($"CUDA device count: {torch.cuda.device_count()}");
        }
    }
}
