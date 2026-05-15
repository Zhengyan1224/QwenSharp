using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen35;

public class Qwen35RotaryEmbedding : nn.Module
{
    private readonly int _dim;
    private readonly double _theta;
    private readonly int[] _mropeSection;
    private readonly Tensor _invFreq;

    public Qwen35RotaryEmbedding(Qwen35Config config, string name = "rotary_emb") : base(name)
    {
        var partialRotaryFactor = config.PartialRotaryFactor;
        _dim = (int)(config.HeadDim * partialRotaryFactor);
        _theta = config.RopeTheta;

        _mropeSection = config.MRopeSection.Length == 3 ? config.MRopeSection : [11, 11, 10];
        
        var invFreq = 1.0f / torch.pow(_theta, torch.arange(0, _dim, 2, dtype: ScalarType.Float32) / _dim);
        _invFreq = invFreq;
        register_buffer("inv_freq", _invFreq, persistent: false);
    }

    public (Tensor cos, Tensor sin) forward(Tensor hiddenStates, Tensor positionIds)
    {
        using var scope = torch.NewDisposeScope();
        
        // positionIds: [bsz, seqLen] or [3, bsz, seqLen]
        if (positionIds.ndim == 2)
        {
            positionIds = positionIds.unsqueeze(0).expand(3, positionIds.shape[0], positionIds.shape[1]);
        }

        var invFreqExpanded = _invFreq.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(3, positionIds.shape[1], -1, 1);
        var positionIdsExpanded = positionIds.unsqueeze(2).to(ScalarType.Float32); // [3, bs, 1, seqLen]

        var freqs = torch.matmul(invFreqExpanded, positionIdsExpanded).transpose(2, 3); // [3, bs, seqLen, dim/2]
        
        var freqsT = ApplyInterleavedMrope(freqs, _mropeSection);
        
        var emb = torch.cat(new[] { freqsT, freqsT }, dim: -1);
        var cos = emb.cos();
        var sin = emb.sin();

        return (scope.MoveToOuter(cos.to(hiddenStates.dtype)), scope.MoveToOuter(sin.to(hiddenStates.dtype)));
    }

    public static (Tensor qEmbed, Tensor kEmbed) ApplyRotaryPosEmb(Tensor q, Tensor k, Tensor cos, Tensor sin, int unsqueezeDim = 1)
    {
        using var scope = torch.NewDisposeScope();
        
        cos = cos.unsqueeze(unsqueezeDim);
        sin = sin.unsqueeze(unsqueezeDim);

        var rotaryDim = cos.shape[^1];
        
        var qRot = q[.., .., .., ..(int)rotaryDim];
        var qPass = q[.., .., .., (int)rotaryDim..];
        
        var kRot = k[.., .., .., ..(int)rotaryDim];
        var kPass = k[.., .., .., (int)rotaryDim..];

        var qEmbed = (qRot * cos) + (RotateHalf(qRot) * sin);
        var kEmbed = (kRot * cos) + (RotateHalf(kRot) * sin);

        qEmbed = torch.cat(new[] { qEmbed, qPass }, dim: -1);
        kEmbed = torch.cat(new[] { kEmbed, kPass }, dim: -1);

        return (scope.MoveToOuter(qEmbed), scope.MoveToOuter(kEmbed));
    }

    private static Tensor RotateHalf(Tensor x)
    {
        var half = x.shape[^1] / 2;
        var x1 = x[.., .., .., ..(int)half];
        var x2 = x[.., .., .., (int)half..];
        return torch.cat(new[] { -x2, x1 }, dim: -1);
    }

    private Tensor ApplyInterleavedMrope(Tensor freqs, int[] mropeSection)
    {
        // freqs: [3, bs, seqLen, dim/2]
        var freqsT = freqs[0].clone();
        
        // C# doesn't support complex strided slices like python `freqs_t[..., slice(offset, length, 3)]` natively easily.
        // We will do it via indexed scatter or chunking.
        var dimHalf = freqsT.shape[^1];
        
        for (int dim = 1; dim <= 2; dim++) // H, W
        {
            var offset = dim;
            var length = mropeSection[dim] * 3;
            
            // Generate indices: offset, offset+3, offset+6 ... length-3
            for (int i = offset; i < length; i += 3)
            {
                if (i < dimHalf)
                {
                    freqsT[.., .., i] = freqs[dim, .., .., i];
                }
            }
        }
        
        return freqsT;
    }
}
