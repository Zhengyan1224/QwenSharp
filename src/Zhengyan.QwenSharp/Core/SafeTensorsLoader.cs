using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.Core;

/// <summary>
/// Loads model weights from HuggingFace .safetensors files.
///
/// The safetensors format:
///   1. 8-byte little-endian uint64 = length of JSON header
///   2. JSON header: tensor name 鈫?{ dtype, shape, data_offsets:[begin, end] }
///   3. Raw binary tensor data
///
/// Supports single-file and sharded (model.safetensors.index.json) weight loading.
/// This loader is used by the pure .NET runtime path for manual weight mapping
/// and diagnostics.
/// </summary>
public static class SafeTensorsLoader
{
    // Maps safetensors dtype strings 鈫?TorchSharp ScalarType
    private static readonly Dictionary<string, ScalarType> DtypeMap = new(StringComparer.OrdinalIgnoreCase)
    {
        { "F64",  ScalarType.Float64  },
        { "F32",  ScalarType.Float32  },
        { "F16",  ScalarType.Float16  },
        { "BF16", ScalarType.BFloat16 },
        { "I64",  ScalarType.Int64    },
        { "I32",  ScalarType.Int32    },
        { "I16",  ScalarType.Int16    },
        { "I8",   ScalarType.Int8     },
        { "U8",   ScalarType.Byte     },
        { "BOOL", ScalarType.Bool     },
    };

    // Bytes per element for each dtype
    private static readonly Dictionary<string, int> ElementSizeMap = new(StringComparer.OrdinalIgnoreCase)
    {
        { "F64",  8 }, { "F32", 4 }, { "F16",  2 }, { "BF16", 2 },
        { "I64",  8 }, { "I32", 4 }, { "I16",  2 }, { "I8",   1 },
        { "U8",   1 }, { "BOOL", 1 },
    };

    /// <summary>
    /// Loads all tensors from a single .safetensors file.
    /// </summary>
    public static Dictionary<string, Tensor> LoadFile(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException("safetensors file not found", path);
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        return ParseStream(stream);
    }

    /// <summary>
    /// Loads weights from a model directory.
    /// Supports model.safetensors (single) and sharded (index.json).
    /// </summary>
    public static Dictionary<string, Tensor> LoadFromDirectory(string modelDirectory)
    {
        var indexPath = Path.Combine(modelDirectory, "model.safetensors.index.json");
        if (File.Exists(indexPath))
            return LoadSharded(modelDirectory, indexPath);

        var singlePath = Path.Combine(modelDirectory, "model.safetensors");
        if (File.Exists(singlePath))
            return LoadFile(singlePath);

        var discoveredSingleFile = Directory.GetFiles(modelDirectory, "*.safetensors")
            .FirstOrDefault(path => !path.EndsWith(".index.json", StringComparison.OrdinalIgnoreCase));
        if (discoveredSingleFile is not null)
            return LoadFile(discoveredSingleFile);

        throw new FileNotFoundException(
            $"No safetensors weights found in {modelDirectory}. " +
            "Expected model.safetensors, any single *.safetensors file, or model.safetensors.index.json.");
    }

    private static Dictionary<string, Tensor> LoadSharded(string modelDirectory, string indexPath)
    {
        var indexJson = File.ReadAllText(indexPath);
        using var doc = JsonDocument.Parse(indexJson);
        var weightMap = doc.RootElement.GetProperty("weight_map");

        var shardSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var prop in weightMap.EnumerateObject())
            shardSet.Add(prop.Value.GetString()!);

        var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var shardFile in shardSet)
        {
            var filePath = Path.Combine(modelDirectory, shardFile);
            foreach (var (name, tensor) in LoadFile(filePath))
                result[name] = tensor;
        }
        return result;
    }

    private static Dictionary<string, Tensor> ParseStream(Stream stream)
    {
        // 1. Read 8-byte header length
        Span<byte> lenBuf = stackalloc byte[8];
        stream.ReadExactly(lenBuf);
        var headerLen = BinaryPrimitives.ReadUInt64LittleEndian(lenBuf);

        if (headerLen > 100 * 1024 * 1024)
            throw new InvalidDataException("safetensors header exceeds 100MB 鈥?likely corrupt.");

        // 2. Read JSON header
        var headerBytes = new byte[headerLen];
        stream.ReadExactly(headerBytes);
        var headerJson = Encoding.UTF8.GetString(headerBytes);

        // Data section starts after 8 + headerLen bytes
        var dataStart = 8L + (long)headerLen;

        using var doc = JsonDocument.Parse(headerJson);
        var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);

        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (prop.Name == "__metadata__") continue;

            var meta    = prop.Value;
            var dtype   = meta.GetProperty("dtype").GetString()!;
            var shape   = ParseShape(meta.GetProperty("shape"));
            var offsets = meta.GetProperty("data_offsets");
            var begin   = offsets[0].GetInt64();
            var end     = offsets[1].GetInt64();

            if (!DtypeMap.TryGetValue(dtype, out var scalarType))
                throw new NotSupportedException($"Unsupported dtype '{dtype}' in tensor '{prop.Name}'");

            var byteCount = end - begin;
            var rawData   = new byte[byteCount];
            stream.Seek(dataStart + begin, SeekOrigin.Begin);
            stream.ReadExactly(rawData);

            result[prop.Name] = CreateTensor(rawData, shape, dtype, scalarType);
        }

        return result;
    }

    private static long[] ParseShape(JsonElement shapeElem)
    {
        var shape = new long[shapeElem.GetArrayLength()];
        int i = 0;
        foreach (var d in shapeElem.EnumerateArray())
            shape[i++] = d.GetInt64();
        return shape;
    }

    /// <summary>
    /// Creates a TorchSharp Tensor from raw bytes.
    /// Allocates an empty native tensor and copies bytes directly to its data pointer.
    /// This supports all dtypes (including FP16/BF16) seamlessly.
    /// </summary>
    private static Tensor CreateTensor(byte[] data, long[] shape, string dtype, ScalarType scalarType)
    {
        var tensor = torch.empty(shape, dtype: scalarType);
        data.AsSpan().CopyTo(tensor.bytes);
        return tensor;
    }
}
