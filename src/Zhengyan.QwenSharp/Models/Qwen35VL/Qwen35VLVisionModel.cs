using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Qwen2VL;
using Zhengyan.QwenSharp.Models.Qwen3VL;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen35VL;

public class Qwen35VLVisionPatchEmbed : nn.Module
{
    private readonly int _patchSize;
    private readonly int _temporalPatchSize;
    private readonly int _inChannels;
    private readonly int _embedDim;
    private readonly Conv3d _proj;

    public Qwen35VLVisionPatchEmbed(Qwen3VLVisionConfig config, string name = "patch_embed") : base(name)
    {
        _patchSize = config.PatchSize;
        _temporalPatchSize = config.TemporalPatchSize;
        _inChannels = config.InChannels;
        _embedDim = config.HiddenSize;

        _proj = Conv3d(_inChannels, _embedDim, kernel_size: (_temporalPatchSize, _patchSize, _patchSize), stride: (_temporalPatchSize, _patchSize, _patchSize), bias: true);
        register_module("proj", _proj);
    }

    public Tensor forward(Tensor hiddenStates)
    {
        using var scope = torch.NewDisposeScope();
        var targetDtype = _proj.weight.dtype;
        hiddenStates = hiddenStates.view(-1, _inChannels, _temporalPatchSize, _patchSize, _patchSize);
        hiddenStates = _proj.forward(hiddenStates.to_type(targetDtype)).view(-1, _embedDim);
        return scope.MoveToOuter(hiddenStates);
    }
}

public class Qwen35VLPatchMerger : nn.Module
{
    private readonly LayerNorm _norm;
    private readonly Linear _linearFc1;
    private readonly Linear _linearFc2;
    private readonly int _hiddenSize;

    public Qwen35VLPatchMerger(int outHiddenSize, int contextDim, int spatialMergeSize, string name) : base(name)
    {
        _hiddenSize = contextDim * spatialMergeSize * spatialMergeSize;
        _norm = LayerNorm(new long[] { contextDim }, eps: 1e-6);
        _linearFc1 = Linear(_hiddenSize, _hiddenSize);
        _linearFc2 = Linear(_hiddenSize, outHiddenSize);

        register_module("norm", _norm);
        register_module("linear_fc1", _linearFc1);
        register_module("linear_fc2", _linearFc2);
    }

    public Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        x = _norm.forward(x).view(-1, _hiddenSize);
        x = _linearFc2.forward(torch.nn.functional.gelu(_linearFc1.forward(x)));
        return scope.MoveToOuter(x);
    }
}

public class Qwen35VLVisionAttention : nn.Module
{
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly Linear _qkv;
    private readonly Linear _proj;

    public Qwen35VLVisionAttention(Qwen3VLVisionConfig config, string name = "attn") : base(name)
    {
        _dim = config.HiddenSize;
        _numHeads = config.NumHeads;
        _headDim = _dim / _numHeads;

        _qkv = Linear(_dim, _dim * 3, hasBias: true);
        _proj = Linear(_dim, _dim, hasBias: true);

        register_module("qkv", _qkv);
        register_module("proj", _proj);
    }

    private Tensor RotateHalf(Tensor x)
    {
        var half = x.shape[^1] / 2;
        var x1 = x.slice(-1, 0, half, 1);
        var x2 = x.slice(-1, half, x.shape[^1], 1);
        return torch.cat(new[] { -x2, x1 }, dim: -1);
    }

    public Tensor forward(Tensor hiddenStates, Tensor cuSeqlens, Tensor cos, Tensor sin)
    {
        using var scope = torch.NewDisposeScope();

        var seqLength = hiddenStates.shape[0];
        var qkvOutput = _qkv.forward(hiddenStates).reshape(seqLength, 3, _numHeads, _headDim).permute(1, 0, 2, 3);

        var q = qkvOutput[0];
        var k = qkvOutput[1];
        var v = qkvOutput[2];

        var qf = q.to_type(ScalarType.Float32);
        var kf = k.to_type(ScalarType.Float32);
        var cosF = cos.unsqueeze(-2).to_type(ScalarType.Float32);
        var sinF = sin.unsqueeze(-2).to_type(ScalarType.Float32);

        q = ((qf * cosF) + (RotateHalf(qf) * sinF)).to_type(hiddenStates.dtype).transpose(0, 1).unsqueeze(0);
        k = ((kf * cosF) + (RotateHalf(kf) * sinF)).to_type(hiddenStates.dtype).transpose(0, 1).unsqueeze(0);
        v = v.transpose(0, 1).unsqueeze(0);

        var cuSeqlensArray = cuSeqlens.dtype == ScalarType.Int32 ? Array.ConvertAll(cuSeqlens.data<int>().ToArray(), static x => (long)x) : cuSeqlens.data<long>().ToArray();
        var attnOutputParts = new List<Tensor>();

        for (int i = 0; i < cuSeqlensArray.Length - 1; i++)
        {
            var len = cuSeqlensArray[i + 1] - cuSeqlensArray[i];
            var qChunk = q.narrow(2, cuSeqlensArray[i], len);
            var kChunk = k.narrow(2, cuSeqlensArray[i], len);
            var vChunk = v.narrow(2, cuSeqlensArray[i], len);
            attnOutputParts.Add(torch.nn.functional.scaled_dot_product_attention(qChunk, kChunk, vChunk, null, 0.0, false).transpose(1, 2));
        }

        var attnOutput = torch.cat(attnOutputParts, dim: 1).reshape(seqLength, -1).contiguous();
        return scope.MoveToOuter(_proj.forward(attnOutput));
    }
}

public class Qwen35VLVisionMLP : nn.Module
{
    private readonly Linear _linearFc1;
    private readonly Linear _linearFc2;
    private readonly string _hiddenAct;

    public Qwen35VLVisionMLP(Qwen3VLVisionConfig config, string name = "mlp") : base(name)
    {
        _linearFc1 = Linear(config.HiddenSize, config.IntermediateSize, hasBias: true);
        _linearFc2 = Linear(config.IntermediateSize, config.HiddenSize, hasBias: true);
        _hiddenAct = config.HiddenAct;

        register_module("linear_fc1", _linearFc1);
        register_module("linear_fc2", _linearFc2);
    }

    private Tensor ApplyActivation(Tensor x)
    {
        if (string.Equals(_hiddenAct, "gelu_pytorch_tanh", StringComparison.OrdinalIgnoreCase))
        {
            using var cubic = x.pow(3);
            using var inner = x + (0.044715 * cubic);
            using var scaled = inner * Math.Sqrt(2.0 / Math.PI);
            return 0.5 * x * (1.0 + scaled.tanh());
        }

        return torch.nn.functional.gelu(x);
    }

    public Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        x = ApplyActivation(_linearFc1.forward(x));
        x = _linearFc2.forward(x);
        return scope.MoveToOuter(x);
    }
}

public class Qwen35VLVisionBlock : nn.Module
{
    private readonly LayerNorm _norm1;
    private readonly LayerNorm _norm2;
    private readonly Qwen35VLVisionAttention _attn;
    private readonly Qwen35VLVisionMLP _mlp;

    public Qwen35VLVisionBlock(Qwen3VLVisionConfig config, string name = "block") : base(name)
    {
        _norm1 = LayerNorm(new long[] { config.HiddenSize }, eps: 1e-6);
        _norm2 = LayerNorm(new long[] { config.HiddenSize }, eps: 1e-6);
        _attn = new Qwen35VLVisionAttention(config, name: "attn");
        _mlp = new Qwen35VLVisionMLP(config, name: "mlp");

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

public class Qwen35VisionTransformer : nn.Module
{
    private readonly int _spatialMergeSize;
    private readonly Qwen35VLVisionPatchEmbed _patchEmbed;
    private readonly Embedding _posEmbed;
    private readonly int _numGridPerSide;
    private readonly VisionRotaryEmbedding _rotaryPosEmb;
    private readonly ModuleList<Qwen35VLVisionBlock> _blocks;
    private readonly Qwen35VLPatchMerger _merger;

    public Qwen35VisionTransformer(Qwen3VLVisionConfig config, string name = "visual") : base(name)
    {
        _spatialMergeSize = config.SpatialMergeSize;

        _patchEmbed = new Qwen35VLVisionPatchEmbed(config, name: "patch_embed");
        _posEmbed = Embedding(config.NumPositionEmbeddings, config.HiddenSize);
        _numGridPerSide = (int)Math.Sqrt(config.NumPositionEmbeddings);
        int headDim = config.HiddenSize / config.NumHeads;
        _rotaryPosEmb = new VisionRotaryEmbedding(headDim / 2, name: "rotary_pos_emb");

        var blocksList = new List<Qwen35VLVisionBlock>();
        for (int i = 0; i < config.Depth; i++)
        {
            blocksList.Add(new Qwen35VLVisionBlock(config, name: $"{i}"));
        }
        _blocks = ModuleList(blocksList.ToArray());
        _merger = new Qwen35VLPatchMerger(config.OutHiddenSize, config.HiddenSize, config.SpatialMergeSize, name: "merger");

        register_module("patch_embed", _patchEmbed);
        register_module("pos_embed", _posEmbed);
        register_module("rotary_pos_emb", _rotaryPosEmb);
        register_module("blocks", _blocks);
        register_module("merger", _merger);
    }

    private Tensor FastPosEmbedInterpolate(Tensor gridThw)
    {
        using var scope = torch.NewDisposeScope();
        var patchEmbedsPerImage = new List<Tensor>();

        for (int i = 0; i < gridThw.shape[0]; i++)
        {
            long t = gridThw[i, 0].item<long>();
            long h = gridThw[i, 1].item<long>();
            long w = gridThw[i, 2].item<long>();

            using var hIdxs = torch.linspace(0, _numGridPerSide - 1, h, device: gridThw.device, dtype: ScalarType.Float32);
            using var wIdxs = torch.linspace(0, _numGridPerSide - 1, w, device: gridThw.device, dtype: ScalarType.Float32);

            using var hFloor = hIdxs.floor().to_type(ScalarType.Int64);
            using var wFloor = wIdxs.floor().to_type(ScalarType.Int64);
            using var hCeil = (hFloor + 1).clamp(max: _numGridPerSide - 1);
            using var wCeil = (wFloor + 1).clamp(max: _numGridPerSide - 1);

            using var dh = hIdxs - hFloor.to_type(ScalarType.Float32);
            using var dw = wIdxs - wFloor.to_type(ScalarType.Float32);

            using var baseH = hFloor * _numGridPerSide;
            using var baseHCeil = hCeil * _numGridPerSide;

            using var idx00 = (baseH.unsqueeze(1) + wFloor.unsqueeze(0)).reshape(-1);
            using var idx01 = (baseH.unsqueeze(1) + wCeil.unsqueeze(0)).reshape(-1);
            using var idx10 = (baseHCeil.unsqueeze(1) + wFloor.unsqueeze(0)).reshape(-1);
            using var idx11 = (baseHCeil.unsqueeze(1) + wCeil.unsqueeze(0)).reshape(-1);

            using var w00 = ((1 - dh).unsqueeze(1) * (1 - dw).unsqueeze(0)).reshape(-1).unsqueeze(1);
            using var w01 = ((1 - dh).unsqueeze(1) * dw.unsqueeze(0)).reshape(-1).unsqueeze(1);
            using var w10 = (dh.unsqueeze(1) * (1 - dw).unsqueeze(0)).reshape(-1).unsqueeze(1);
            using var w11 = (dh.unsqueeze(1) * dw.unsqueeze(0)).reshape(-1).unsqueeze(1);

            using var pos00 = _posEmbed.forward(idx00) * w00;
            using var pos01 = _posEmbed.forward(idx01) * w01;
            using var pos10 = _posEmbed.forward(idx10) * w10;
            using var pos11 = _posEmbed.forward(idx11) * w11;
            var posEmbed = pos00 + pos01 + pos10 + pos11;

            if (t > 1)
            {
                posEmbed = posEmbed.repeat(new long[] { t, 1 });
            }

            posEmbed = posEmbed
                .view(t, h / _spatialMergeSize, _spatialMergeSize, w / _spatialMergeSize, _spatialMergeSize, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4);

            patchEmbedsPerImage.Add(scope.MoveToOuter(posEmbed));
        }

        return scope.MoveToOuter(torch.cat(patchEmbedsPerImage, dim: 0));
    }

    private Tensor RotPosEmb(Tensor gridThw)
    {
        using var scope = torch.NewDisposeScope();
        var posIdsList = new List<Tensor>();
        int numImages = (int)gridThw.shape[0];

        for (int i = 0; i < numImages; i++)
        {
            long t = gridThw[i, 0].item<long>();
            long h = gridThw[i, 1].item<long>();
            long w = gridThw[i, 2].item<long>();

            using var hposIdsBase = torch.arange(h, device: gridThw.device).unsqueeze(1).expand(h, w);
            var hposIds = hposIdsBase.reshape(h / _spatialMergeSize, _spatialMergeSize, w / _spatialMergeSize, _spatialMergeSize);
            hposIds = hposIds.permute(0, 2, 1, 3).flatten();

            using var wposIdsBase = torch.arange(w, device: gridThw.device).unsqueeze(0).expand(h, w);
            var wposIds = wposIdsBase.reshape(h / _spatialMergeSize, _spatialMergeSize, w / _spatialMergeSize, _spatialMergeSize);
            wposIds = wposIds.permute(0, 2, 1, 3).flatten();

            using var stacked = torch.stack(new Tensor[] { hposIds, wposIds }, dim: -1);
            posIdsList.Add(scope.MoveToOuter(stacked.repeat(new long[] { t, 1 })));
        }

        using var posIds = torch.cat(posIdsList, dim: 0);
        var maxGridSize = gridThw[.., 1..].max().item<long>();
        using var rotaryPosEmbFull = _rotaryPosEmb.forward(maxGridSize);
        using var hPosEmb = rotaryPosEmbFull.index_select(0, posIds[.., 0]);
        using var wPosEmb = rotaryPosEmbFull.index_select(0, posIds[.., 1]);
        return scope.MoveToOuter(torch.cat(new Tensor[] { hPosEmb, wPosEmb }, dim: -1).flatten(1));
    }

    public Tensor forward(Tensor hiddenStates, Tensor gridThw)
    {
        using var scope = torch.NewDisposeScope();
        hiddenStates = _patchEmbed.forward(hiddenStates);
        hiddenStates = hiddenStates + FastPosEmbedInterpolate(gridThw);

        using var rotaryPosEmb = RotPosEmb(gridThw);
        using var emb = torch.cat(new[] { rotaryPosEmb, rotaryPosEmb }, dim: -1);
        using var cos = emb.cos();
        using var sin = emb.sin();
        using var patchCounts = gridThw[.., 1] * gridThw[.., 2];
        using var patchCountsRep = patchCounts.repeat_interleave(gridThw[.., 0]);
        var cuSeqlens = patchCountsRep.to_type(ScalarType.Int32).cumsum(dim: 0);
        cuSeqlens = torch.nn.functional.pad(cuSeqlens, new long[] { 1, 0 }, value: 0);

        for (int layerIdx = 0; layerIdx < _blocks.Count; layerIdx++)
        {
            hiddenStates = _blocks[layerIdx].forward(hiddenStates, cuSeqlens, cos, sin);
        }

        return scope.MoveToOuter(_merger.forward(hiddenStates));
    }
}
