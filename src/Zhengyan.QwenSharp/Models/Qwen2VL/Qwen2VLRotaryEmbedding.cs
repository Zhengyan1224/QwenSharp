using System;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.Models.Qwen2VL;

public class Qwen2VLRotaryEmbedding : nn.Module
{
    private readonly int _maxSeqLenCached;
    private readonly double _ropeTheta;
    private readonly Tensor _invFreq;

    public Qwen2VLRotaryEmbedding(Qwen2VLConfig config, string name = "rotary_emb") : base(name)
    {
        _maxSeqLenCached = config.MaxPositionEmbeddings;
        _ropeTheta = config.RopeTheta;

        int dim = (int)(config.HiddenSize / config.NumAttentionHeads);
        
        var seq = torch.arange(0, dim, 2, dtype: ScalarType.Float32);
        _invFreq = 1.0 / torch.pow(_ropeTheta, seq / dim);

        register_buffer("inv_freq", _invFreq, persistent: false);
    }

    public (Tensor Cos, Tensor Sin) forward(Tensor x, Tensor positionIds)
    {
        using var scope = torch.NewDisposeScope();

        // positionIds shape is expected to be [3, bs, seq_len] for Qwen2-VL
        // inv_freq -> [3, seq_len, dim/2, 1]
        var invFreqExpanded = _invFreq.unsqueeze(0).unsqueeze(0).unsqueeze(3).expand(3, positionIds.shape[1], -1, 1);
        
        // positionIdsExpanded -> [3, bs, 1, seq_len]
        var positionIdsExpanded = positionIds.unsqueeze(2).to_type(ScalarType.Float32);
        
        // freqs -> [3, bs, seq_len, dim/2]
        var freqs = torch.matmul(invFreqExpanded, positionIdsExpanded).transpose(2, 3);
        
        var emb = torch.cat(new[] { freqs, freqs }, dim: -1);
        var cos = emb.cos();
        var sin = emb.sin();

        return (scope.MoveToOuter(cos.to_type(x.dtype)), scope.MoveToOuter(sin.to_type(x.dtype)));
    }

    public static (Tensor qEmbed, Tensor kEmbed) ApplyMultimodalRotaryPosEmb(Tensor q, Tensor k, Tensor cos, Tensor sin, int[] mropeSection, int unsqueezeDim = 1)
    {
        using var scope = torch.NewDisposeScope();
        
        // mropeSection is roughly [16, 24, 24] for a 128 head_dim, representing half channels for T, H, W.
        // Multiply by 2 because we concatenate freqs, freqs.
        var fullSections = mropeSection.Select(s => s * 2).ToArray();

        // cos shape: [3, bs, seq_len, head_dim]
        // We split it along the last dimension into 3 chunks according to fullSections
        var cosSplits = cos.split(fullSections.Select(s => (long)s).ToArray(), dim: -1);
        var sinSplits = sin.split(fullSections.Select(s => (long)s).ToArray(), dim: -1);

        var cosParts = new System.Collections.Generic.List<Tensor>();
        var sinParts = new System.Collections.Generic.List<Tensor>();

        for (int i = 0; i < fullSections.Length; i++)
        {
            // The split arrays have length 3 (temporal, height, width).
            // cosSplits[i] has shape [3, bs, seq_len, section_dim]
            // We select index (i % 3) from the 0-th dimension.
            int idx = i % 3;
            cosParts.Add(cosSplits[i][idx]);
            sinParts.Add(sinSplits[i][idx]);
        }

        var mCos = torch.cat(cosParts, dim: -1).unsqueeze(unsqueezeDim);
        var mSin = torch.cat(sinParts, dim: -1).unsqueeze(unsqueezeDim);

        var origQDtype = q.dtype;
        var origKDtype = k.dtype;

        q = q.to_type(ScalarType.Float32);
        k = k.to_type(ScalarType.Float32);

        var qEmbed = (q * mCos) + (RotateHalf(q) * mSin);
        var kEmbed = (k * mCos) + (RotateHalf(k) * mSin);

        return (scope.MoveToOuter(qEmbed.to_type(origQDtype)), scope.MoveToOuter(kEmbed.to_type(origKDtype)));
    }

    private static Tensor RotateHalf(Tensor x)
    {
        var half = x.shape[^1] / 2;
        var x1 = x[.., .., .., ..(int)half];
        var x2 = x[.., .., .., (int)half..];
        return torch.cat(new[] { -x2, x1 }, dim: -1);
    }
}
