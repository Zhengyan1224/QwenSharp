using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen2VL;

public class PatchEmbed : nn.Module
{
    private readonly int _patchSize;
    private readonly int _temporalPatchSize;
    private readonly int _inChannels;
    private readonly int _embedDim;
    private readonly Conv3d _proj;

    public PatchEmbed(Qwen2VLVisionConfig config, string name = "patch_embed") : base(name)
    {
        _patchSize = config.PatchSize;
        _temporalPatchSize = config.TemporalPatchSize;
        _inChannels = config.InChannels;
        _embedDim = config.EmbedDim;

        _proj = Conv3d(_inChannels, _embedDim, kernel_size: (_temporalPatchSize, _patchSize, _patchSize), stride: (_temporalPatchSize, _patchSize, _patchSize), bias: false);
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

public class VisionRotaryEmbedding : nn.Module
{
    private readonly int _dim;
    private readonly double _theta;
    private readonly Tensor _invFreq;

    public VisionRotaryEmbedding(int dim, double theta = 10000.0, string name = "rotary_pos_emb") : base(name)
    {
        _dim = dim;
        _theta = theta;
        
        var seq = torch.arange(0, dim, 2, dtype: ScalarType.Float32);
        _invFreq = 1.0 / torch.pow(_theta, seq / dim);
        register_buffer("inv_freq", _invFreq, persistent: false);
    }

    public Tensor forward(long seqLen)
    {
        using var scope = torch.NewDisposeScope();
        
        var seq = torch.arange(seqLen, device: _invFreq.device, dtype: _invFreq.dtype);
        var freqs = torch.outer(seq, _invFreq);
        
        return scope.MoveToOuter(freqs);
    }
}

public class PatchMerger : nn.Module
{
    private readonly int _hiddenSize;
    private readonly LayerNorm _lnQ;
    private readonly Sequential _mlp;

    public PatchMerger(int dim, int contextDim, int spatialMergeSize = 2, string name = "merger") : base(name)
    {
        _hiddenSize = contextDim * (spatialMergeSize * spatialMergeSize);
        _lnQ = LayerNorm(new long[] { contextDim }, eps: 1e-6);
        
        _mlp = Sequential(
            Linear(_hiddenSize, _hiddenSize),
            GELU(),
            Linear(_hiddenSize, dim)
        );

        register_module("ln_q", _lnQ);
        register_module("mlp", _mlp);
    }

    public Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        
        x = _mlp.forward(_lnQ.forward(x).view(-1, _hiddenSize));
        
        return scope.MoveToOuter(x);
    }
}

public class VisionMlp : nn.Module
{
    private readonly Linear _fc1;
    private readonly nn.Module<Tensor, Tensor> _act;
    private readonly Linear _fc2;

    public VisionMlp(int dim, int hiddenDim, string hiddenAct, string name = "mlp") : base(name)
    {
        _fc1 = Linear(dim, hiddenDim);
        
        // Match Python ACT2FN logic
        _act = hiddenAct switch
        {
            "gelu" => GELU(),
            "quick_gelu" => GELU(), // Approximate for now, quick_gelu is x * sigmoid(1.702 * x)
            "silu" => SiLU(),
            _ => GELU()
        };
        
        _fc2 = Linear(hiddenDim, dim);

        register_module("fc1", _fc1);
        register_module("act", _act);
        register_module("fc2", _fc2);
    }

    public Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        var t = _act.forward(_fc1.forward(x));
        return scope.MoveToOuter(_fc2.forward(t));
    }
}

public class Qwen2VisionTransformer : nn.Module
{
    private readonly int _spatialMergeSize;
    private readonly PatchEmbed _patchEmbed;
    private readonly VisionRotaryEmbedding _rotaryPosEmb;
    private readonly ModuleList<Qwen2VLVisionBlock> _blocks;
    private readonly PatchMerger _merger;

    public Qwen2VisionTransformer(Qwen2VLVisionConfig config, string name = "visual") : base(name)
    {
        _spatialMergeSize = config.SpatialMergeSize;
        
        _patchEmbed = new PatchEmbed(config, name: "patch_embed");
        
        int headDim = config.EmbedDim / config.NumHeads;
        _rotaryPosEmb = new VisionRotaryEmbedding(headDim / 2, name: "rotary_pos_emb");
        
        var blocksList = new System.Collections.Generic.List<Qwen2VLVisionBlock>();
        for (int i = 0; i < config.Depth; i++)
        {
            blocksList.Add(new Qwen2VLVisionBlock(config, name: $"{i}"));
        }
        _blocks = ModuleList(blocksList.ToArray());
        
        _merger = new PatchMerger(config.HiddenSize, config.EmbedDim, _spatialMergeSize, name: "merger");

        register_module("patch_embed", _patchEmbed);
        register_module("rotary_pos_emb", _rotaryPosEmb);
        register_module("blocks", _blocks);
        register_module("merger", _merger);
    }

    private Tensor GetRotPosEmb(Tensor gridThw)
    {
        using var scope = torch.NewDisposeScope();
        int numImages = (int)gridThw.shape[0];
        
        var posIdsList = new System.Collections.Generic.List<Tensor>();
        for (int i = 0; i < numImages; i++)
        {
            long t = gridThw[i, 0].item<long>();
            long h = gridThw[i, 1].item<long>();
            long w = gridThw[i, 2].item<long>();

            var hposIds = torch.arange(h).unsqueeze(1).expand(h, w);
            hposIds = hposIds.reshape(
                h / _spatialMergeSize, _spatialMergeSize,
                w / _spatialMergeSize, _spatialMergeSize
            );
            hposIds = hposIds.permute(0, 2, 1, 3).flatten();

            var wposIds = torch.arange(w).unsqueeze(0).expand(h, w);
            wposIds = wposIds.reshape(
                h / _spatialMergeSize, _spatialMergeSize,
                w / _spatialMergeSize, _spatialMergeSize
            );
            wposIds = wposIds.permute(0, 2, 1, 3).flatten();
            
            var stacked = torch.stack(new Tensor[] { hposIds, wposIds }, dim: -1);
            posIdsList.Add(stacked.repeat(new long[] { t, 1 }));
        }

        var posIds = torch.cat(posIdsList, dim: 0);
        
        var maxGridSize = gridThw[.., 1..].max().item<long>();
        var rotaryPosEmbFull = _rotaryPosEmb.forward(maxGridSize);
        
        var hPosEmb = rotaryPosEmbFull.index_select(0, posIds[.., 0]);
        var wPosEmb = rotaryPosEmbFull.index_select(0, posIds[.., 1]);
        
        var rotaryPosEmb = torch.cat(new Tensor[] { hPosEmb, wPosEmb }, dim: -1);
        
        return scope.MoveToOuter(rotaryPosEmb);
    }

    public Tensor forward(Tensor hiddenStates, Tensor gridThw)
    {
        using var scope = torch.NewDisposeScope();
        
        hiddenStates = _patchEmbed.forward(hiddenStates);
        var rotaryPosEmb = GetRotPosEmb(gridThw);
        
        var emb = torch.cat(new Tensor[] { rotaryPosEmb, rotaryPosEmb }, dim: -1);
        var cos = emb.cos();
        var sin = emb.sin();

        var patchCounts = gridThw[.., 1] * gridThw[.., 2];
        var patchCountsRep = patchCounts.repeat_interleave(gridThw[.., 0]);
        
        var cuSeqlens = patchCountsRep.to_type(ScalarType.Int32).cumsum(dim: 0);
        cuSeqlens = torch.nn.functional.pad(cuSeqlens, new long[] { 1, 0 }, value: 0);

        foreach (var block in _blocks)
        {
            hiddenStates = block.forward(hiddenStates, cuSeqlens, cos, sin);
        }

        var mergedHiddenStates = _merger.forward(hiddenStates);
        return scope.MoveToOuter(mergedHiddenStates);
    }
}
