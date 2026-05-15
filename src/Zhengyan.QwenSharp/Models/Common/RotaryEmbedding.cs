using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.Models.Common;

/// <summary>
/// RoPE (Rotary Position Embedding).
/// Extracts cos and sin embeddings based on position_ids to apply rotary embeddings.
/// </summary>
public class RotaryEmbedding : nn.Module<Tensor, Tensor, (Tensor cos, Tensor sin)>
{
    private readonly int _dim;
    private readonly double _base;
    private readonly int[]? _mropeSection;
    private readonly Tensor _invFreq;
    private long _seqLenCached;
    private Tensor _cosCached = null!;
    private Tensor _sinCached = null!;

    public RotaryEmbedding(
        int dim,
        int maxPositionEmbeddings = 32768,
        double baseValue = 10000.0,
        int[]? mropeSection = null,
        string name = "RotaryEmbedding") : base(name)
    {
        _dim = dim;
        _base = baseValue;
        _mropeSection = mropeSection;

        // inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        var it = torch.arange(0, dim, 2, dtype: ScalarType.Float32);
        _invFreq = 1.0 / torch.pow(_base, it / dim);
        register_buffer("inv_freq", _invFreq, persistent: false);

        UpdateCachedTensors(maxPositionEmbeddings);
        
        RegisterComponents();
    }

    private void UpdateCachedTensors(long seqLen)
    {
        _seqLenCached = seqLen;

        using var scope = torch.NewDisposeScope();
        
        // t = torch.arange(seq_len, device=device, dtype=torch.float32)
        var t = torch.arange(seqLen, device: _invFreq.device, dtype: _invFreq.dtype);
        
        // freqs = torch.outer(t, inv_freq)
        var freqs = torch.outer(t, _invFreq);
        
        // emb = torch.cat((freqs, freqs), dim=-1)
        var emb = torch.cat(new[] { freqs, freqs }, dim: -1);
        
        _cosCached?.Dispose();
        _sinCached?.Dispose();
        
        _cosCached = emb.cos().detach();
        _sinCached = emb.sin().detach();
        
        scope.MoveToOuter(_cosCached, _sinCached);
    }

    public override (Tensor cos, Tensor sin) forward(Tensor x, Tensor positionIds)
    {
        // Dynamic update of cached tensors if seq_len is larger than cache
        var maxSeqLen = positionIds.max().item<long>() + 1;
        if (maxSeqLen > _seqLenCached)
        {
            UpdateCachedTensors(maxSeqLen);
        }

        using var scope = torch.NewDisposeScope();

        if (_mropeSection is not null)
        {
            if (positionIds.ndim == 2)
            {
                positionIds = positionIds
                    .unsqueeze(0)
                    .expand(3, positionIds.shape[0], positionIds.shape[1]);
            }

            if (positionIds.ndim != 3)
            {
                throw new ArgumentException("Multimodal RoPE expects positionIds with shape [batch, seq] or [3, batch, seq].");
            }

            var batchSize = positionIds.shape[1];
            var multimodalSeqLen = positionIds.shape[2];
            var multimodalFlatPos = positionIds.flatten();

            var cos3d = _cosCached.index_select(0, multimodalFlatPos).view(3, batchSize, multimodalSeqLen, _cosCached.shape[1]);
            var sin3d = _sinCached.index_select(0, multimodalFlatPos).view(3, batchSize, multimodalSeqLen, _sinCached.shape[1]);
            var (cosMrope, sinMrope) = BuildMultimodalSections(cos3d, sin3d, _mropeSection);

            cosMrope = cosMrope.unsqueeze(1).to_type(x.dtype);
            sinMrope = sinMrope.unsqueeze(1).to_type(x.dtype);

            return (scope.MoveToOuter(cosMrope), scope.MoveToOuter(sinMrope));
        }

        if (positionIds.ndim == 3)
        {
            positionIds = positionIds[0];
        }
        
        var bsz = positionIds.shape[0];
        var seqLen = positionIds.shape[1];
        var flatPos = positionIds.flatten();

        // Gather cos and sin based on positionIds
        var cos = _cosCached.index_select(0, flatPos).view(bsz, seqLen, _cosCached.shape[1]).unsqueeze(1); // [bsz, 1, seq_len, head_dim]
        var sin = _sinCached.index_select(0, flatPos).view(bsz, seqLen, _sinCached.shape[1]).unsqueeze(1);
        
        // Cast to x's dtype
        cos = cos.to_type(x.dtype);
        sin = sin.to_type(x.dtype);

        return (scope.MoveToOuter(cos), scope.MoveToOuter(sin));
    }

    private static (Tensor cos, Tensor sin) BuildMultimodalSections(Tensor cos, Tensor sin, int[] mropeSection)
    {
        var fullSections = mropeSection.Select(section => (long)section * 2).ToArray();
        if (fullSections.Sum() != cos.shape[^1])
        {
            throw new InvalidOperationException(
                $"mrope_section [{string.Join(", ", mropeSection)}] does not match RoPE head dimension {cos.shape[^1]}.");
        }

        var cosSplits = cos.split(fullSections, dim: -1);
        var sinSplits = sin.split(fullSections, dim: -1);
        var cosParts = new List<Tensor>(fullSections.Length);
        var sinParts = new List<Tensor>(fullSections.Length);

        for (var i = 0; i < fullSections.Length; i++)
        {
            var axis = i % 3;
            cosParts.Add(cosSplits[i][axis]);
            sinParts.Add(sinSplits[i][axis]);
        }

        return (torch.cat(cosParts, dim: -1), torch.cat(sinParts, dim: -1));
    }

    /// <summary>
    /// Applies the rotary position embedding to query and key tensors.
    /// q, k: [bsz, num_heads, seq_len, head_dim]
    /// </summary>
    public static (Tensor qEmbed, Tensor kEmbed) ApplyRotaryPosEmb(Tensor q, Tensor k, Tensor cos, Tensor sin)
    {
        using var scope = torch.NewDisposeScope();
        
        var qEmbed = (q * cos) + (RotateHalf(q) * sin);
        var kEmbed = (k * cos) + (RotateHalf(k) * sin);
        
        return (scope.MoveToOuter(qEmbed), scope.MoveToOuter(kEmbed));
    }

    private static Tensor RotateHalf(Tensor x)
    {
        var half = x.shape[^1] / 2;
        var x1 = x.slice(-1, 0, half, 1);
        var x2 = x.slice(-1, half, x.shape[^1], 1);
        return torch.cat(new[] { -x2, x1 }, dim: -1);
    }
}
