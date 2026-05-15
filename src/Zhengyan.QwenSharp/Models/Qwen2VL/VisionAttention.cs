using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen2VL;

public class VisionAttention : nn.Module
{
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly Linear _qkv;
    private readonly Linear _proj;

    public VisionAttention(Qwen2VLVisionConfig config, string name = "attn") : base(name)
    {
        _dim = config.EmbedDim;
        _numHeads = config.NumHeads;
        _headDim = _dim / _numHeads;

        _qkv = Linear(_dim, _dim * 3, hasBias: true);
        _proj = Linear(_dim, _dim);

        register_module("qkv", _qkv);
        register_module("proj", _proj);
    }

    public Tensor forward(Tensor hiddenStates, Tensor cuSeqlens, Tensor cos, Tensor sin)
    {
        using var scope = torch.NewDisposeScope();

        long seqLength = hiddenStates.shape[0];
        var qkv = _qkv.forward(hiddenStates)
            .reshape(seqLength, 3, _numHeads, _headDim)
            .permute(1, 0, 2, 3);

        var chunks = qkv.unbind(0);
        var queryStates = chunks[0];
        var keyStates = chunks[1];
        var valueStates = chunks[2];

        (queryStates, keyStates) = ApplyRotaryPosEmbVision(queryStates, keyStates, cos, sin);

        var lengthsNative = cuSeqlens.slice(0, 1, cuSeqlens.shape[0], 1) - cuSeqlens.slice(0, 0, cuSeqlens.shape[0] - 1, 1);
        var lengthsLong = lengthsNative.data<int>().ToArray().Select(x => (long)x).ToArray();

        var qT = queryStates.transpose(0, 1).unsqueeze(0);
        var kT = keyStates.transpose(0, 1).unsqueeze(0);
        var vT = valueStates.transpose(0, 1).unsqueeze(0);

        var qSplits = qT.split(lengthsLong, dim: 2);
        var kSplits = kT.split(lengthsLong, dim: 2);
        var vSplits = vT.split(lengthsLong, dim: 2);

        var attnOutputParts = new System.Collections.Generic.List<Tensor>();
        for (int i = 0; i < lengthsLong.Length; i++)
        {
            var part = torch.nn.functional.scaled_dot_product_attention(
                qSplits[i], kSplits[i], vSplits[i], null, 0.0, false);
            attnOutputParts.Add(part);
        }

        var catted = torch.cat(attnOutputParts, dim: 2);
        var attnOutput = catted.transpose(1, 2).reshape(seqLength, _dim).contiguous();
        attnOutput = _proj.forward(attnOutput);
        return scope.MoveToOuter(attnOutput);
    }

    public static (Tensor qEmbed, Tensor kEmbed) ApplyRotaryPosEmbVision(Tensor q, Tensor k, Tensor cos, Tensor sin)
    {
        var origQDtype = q.dtype;
        var origKDtype = k.dtype;

        q = q.to_type(ScalarType.Float32);
        k = k.to_type(ScalarType.Float32);
        cos = cos.unsqueeze(-2).to_type(ScalarType.Float32);
        sin = sin.unsqueeze(-2).to_type(ScalarType.Float32);

        var qEmbed = (q * cos) + (RotateHalf(q) * sin);
        var kEmbed = (k * cos) + (RotateHalf(k) * sin);

        return (qEmbed.to_type(origQDtype), kEmbed.to_type(origKDtype));
    }

    private static Tensor RotateHalf(Tensor x)
    {
        var half = x.shape[^1] / 2;
        var x1 = x[.., .., ..(int)half];
        var x2 = x[.., .., (int)half..];
        return torch.cat(new[] { -x2, x1 }, dim: -1);
    }
}

public class Qwen2VLVisionBlock : nn.Module
{
    private readonly LayerNorm _norm1;
    private readonly LayerNorm _norm2;
    private readonly VisionAttention _attn;
    private readonly VisionMlp _mlp;

    public Qwen2VLVisionBlock(Qwen2VLVisionConfig config, string name = "block") : base(name)
    {
        _norm1 = LayerNorm(new long[] { config.EmbedDim }, eps: 1e-6);
        _norm2 = LayerNorm(new long[] { config.EmbedDim }, eps: 1e-6);

        int mlpHiddenDim = (int)(config.EmbedDim * config.MlpRatio);

        _attn = new VisionAttention(config, name: "attn");
        _mlp = new VisionMlp(config.EmbedDim, mlpHiddenDim, config.HiddenAct, name: "mlp");

        register_module("norm1", _norm1);
        register_module("norm2", _norm2);
        register_module("attn", _attn);
        register_module("mlp", _mlp);
    }

    public Tensor forward(Tensor hiddenStates, Tensor cuSeqlens, Tensor cos, Tensor sin)
    {
        using var scope = torch.NewDisposeScope();

        var residual = hiddenStates;
        hiddenStates = _norm1.forward(hiddenStates);
        hiddenStates = residual + _attn.forward(hiddenStates, cuSeqlens, cos, sin);

        residual = hiddenStates;
        hiddenStates = _norm2.forward(hiddenStates);
        hiddenStates = residual + _mlp.forward(hiddenStates);

        return scope.MoveToOuter(hiddenStates);
    }
}
