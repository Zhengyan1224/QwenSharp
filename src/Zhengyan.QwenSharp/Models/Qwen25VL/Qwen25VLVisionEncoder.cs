using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen25VL;

public class Qwen25VLVisionAttention : nn.Module
{
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly Linear _qkv;
    private readonly Linear _proj;
    private readonly double _scaling;

    public Qwen25VLVisionAttention(Qwen25VLVisionConfig config, string name = "attn") : base(name)
    {
        _dim = config.HiddenSize;
        _numHeads = config.NumHeads;
        _headDim = _dim / _numHeads;
        
        _qkv = Linear(_dim, _dim * 3, hasBias: true);
        _proj = Linear(_dim, _dim);
        _scaling = System.Math.Pow(_headDim, -0.5);

        register_module("qkv", _qkv);
        register_module("proj", _proj);
    }

    private Tensor ApplyRotaryPosEmbVision(Tensor q, Tensor k, Tensor cos, Tensor sin)
    {
        var origQDtype = q.dtype;
        var origKDtype = k.dtype;
        
        q = q.to_type(ScalarType.Float32);
        k = k.to_type(ScalarType.Float32);
        
        // cos, sin in python: unsqueeze(-2) -> broadcast to head_dim
        cos = cos.unsqueeze(-2).to_type(ScalarType.Float32);
        sin = sin.unsqueeze(-2).to_type(ScalarType.Float32);

        var qEmbed = (q * cos) + (RotateHalf(q) * sin);
        var kEmbed = (k * cos) + (RotateHalf(k) * sin);

        return torch.cat(new[] { qEmbed.to_type(origQDtype), kEmbed.to_type(origKDtype) }, dim: 0); // Need to unpack
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
        
        // qkv shape: [seq_length, dim * 3] -> [seq_length, 3, num_heads, head_dim] -> permute(1, 0, 2, 3) -> [3, seq_length, num_heads, head_dim]
        var qkvOutput = _qkv.forward(hiddenStates).reshape(seqLength, 3, _numHeads, _headDim).permute(1, 0, 2, 3);
        
        var q = qkvOutput[0];
        var k = qkvOutput[1];
        var v = qkvOutput[2];

        // Apply RoPE
        var origQDtype = q.dtype;
        var qf = q.to_type(ScalarType.Float32);
        var kf = k.to_type(ScalarType.Float32);
        var cosF = cos.unsqueeze(-2).to_type(ScalarType.Float32);
        var sinF = sin.unsqueeze(-2).to_type(ScalarType.Float32);
        
        var qEmbed = (qf * cosF) + (RotateHalf(qf) * sinF);
        var kEmbed = (kf * cosF) + (RotateHalf(kf) * sinF);
        
        q = qEmbed.to_type(origQDtype).transpose(0, 1).unsqueeze(0); // [1, num_heads, seq_length, head_dim]
        k = kEmbed.to_type(origQDtype).transpose(0, 1).unsqueeze(0);
        v = v.transpose(0, 1).unsqueeze(0);

        // Splitting manually like PyTorch implementation does for NaViT block processing
        // lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        var cuSeqlensArray = cuSeqlens.dtype == ScalarType.Int32 ? Array.ConvertAll(cuSeqlens.data<int>().ToArray(), static x => (long)x) : cuSeqlens.data<long>().ToArray();
        var attnOutputParts = new List<Tensor>();

        for (int i = 0; i < cuSeqlensArray.Length - 1; i++)
        {
            var len = cuSeqlensArray[i + 1] - cuSeqlensArray[i];
            
            // Extract chunk from sequence dimension
            // shape of q: [1, num_heads, seq_length, head_dim] - wait! 
            // The split in python is on dim=2 (seq_length).
            var qChunk = q.narrow(2, cuSeqlensArray[i], len);
            var kChunk = k.narrow(2, cuSeqlensArray[i], len);
            var vChunk = v.narrow(2, cuSeqlensArray[i], len);
            
            var attnPart = torch.nn.functional.scaled_dot_product_attention(qChunk, kChunk, vChunk, null, 0.0, false).transpose(1, 2);
            attnOutputParts.Add(attnPart);
        }

        var attnOutput = torch.cat(attnOutputParts, dim: 1);
        attnOutput = attnOutput.reshape(seqLength, -1).contiguous();
        var output = _proj.forward(attnOutput);

        return scope.MoveToOuter(output);
    }
}

public class Qwen25VLPatchMerger : nn.Module
{
    private readonly int _hiddenSize;
    private readonly RMSNorm _lnQ;
    private readonly Sequential _mlp;

    public Qwen25VLPatchMerger(int dim, int contextDim, int spatialMergeSize = 2, string name = "merger") : base(name)
    {
        _hiddenSize = contextDim * (spatialMergeSize * spatialMergeSize);
        _lnQ = new RMSNorm(contextDim, 1e-6, name: "ln_q");
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

public class Qwen25VLVisionMLP : nn.Module
{
    private readonly Linear _gateProj;
    private readonly Linear _upProj;
    private readonly Linear _downProj;
    private readonly nn.Module<Tensor, Tensor> _act;

    public Qwen25VLVisionMLP(Qwen25VLVisionConfig config, string name = "mlp") : base(name)
    {
        _gateProj = Linear(config.HiddenSize, config.IntermediateSize, hasBias: true);
        _upProj = Linear(config.HiddenSize, config.IntermediateSize, hasBias: true);
        _downProj = Linear(config.IntermediateSize, config.HiddenSize, hasBias: true);
        
        _act = config.HiddenAct switch
        {
            "silu" => SiLU(),
            "gelu" => GELU(),
            _ => SiLU()
        };

        register_module("gate_proj", _gateProj);
        register_module("up_proj", _upProj);
        register_module("down_proj", _downProj);
        register_module("act", _act);
    }

    public Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        return scope.MoveToOuter(_downProj.forward(_act.forward(_gateProj.forward(x)) * _upProj.forward(x)));
    }
}

public class Qwen25VLVisionBlock : nn.Module
{
    private readonly RMSNorm _norm1;
    private readonly RMSNorm _norm2;
    private readonly Qwen25VLVisionAttention _attn;
    private readonly Qwen25VLVisionMLP _mlp;

    public Qwen25VLVisionBlock(Qwen25VLVisionConfig config, string name = "block") : base(name)
    {
        _norm1 = new RMSNorm(config.HiddenSize, 1e-6, name: "norm1");
        _norm2 = new RMSNorm(config.HiddenSize, 1e-6, name: "norm2");
        _attn = new Qwen25VLVisionAttention(config, name: "attn");
        _mlp = new Qwen25VLVisionMLP(config, name: "mlp");

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

public class Qwen25VisionTransformer : nn.Module
{
    private readonly int _spatialMergeSize;
    private readonly int _patchSize;
    private readonly int _spatialMergeUnit;
    private readonly int _windowSize;
    private readonly int[] _fullAttBlockIndexes;
    
    // We reuse PatchEmbed and PatchMerger from Qwen2VL
    private readonly Zhengyan.QwenSharp.Models.Qwen2VL.PatchEmbed _patchEmbed;
    private readonly Zhengyan.QwenSharp.Models.Qwen2VL.VisionRotaryEmbedding _rotaryPosEmb;
    private readonly ModuleList<Qwen25VLVisionBlock> _blocks;
    private readonly Qwen25VLPatchMerger _merger;

    public Qwen25VisionTransformer(Qwen25VLVisionConfig config, string name = "visual") : base(name)
    {
        _spatialMergeSize = config.SpatialMergeSize;
        _patchSize = config.PatchSize;
        _fullAttBlockIndexes = config.FullattBlockIndexes;
        _windowSize = config.WindowSize;
        _spatialMergeUnit = _spatialMergeSize * _spatialMergeSize;
        
        // Use an adapter for Qwen2VLVisionConfig since we share PatchEmbed and PatchMerger
        var qwen2VLConfig = new Zhengyan.QwenSharp.Models.Qwen2VL.Qwen2VLVisionConfig
        {
            PatchSize = config.PatchSize,
            TemporalPatchSize = config.TemporalPatchSize,
            InChannels = config.InChannels,
            EmbedDim = config.HiddenSize,
            HiddenSize = config.HiddenSize
        };
        
        _patchEmbed = new Zhengyan.QwenSharp.Models.Qwen2VL.PatchEmbed(qwen2VLConfig, name: "patch_embed");
        
        int headDim = config.HiddenSize / config.NumHeads;
        _rotaryPosEmb = new Zhengyan.QwenSharp.Models.Qwen2VL.VisionRotaryEmbedding(headDim / 2, name: "rotary_pos_emb");
        
        var blocksList = new List<Qwen25VLVisionBlock>();
        for (int i = 0; i < config.Depth; i++)
        {
            blocksList.Add(new Qwen25VLVisionBlock(config, name: $"{i}"));
        }
        _blocks = ModuleList(blocksList.ToArray());
        
        _merger = new Qwen25VLPatchMerger(config.OutHiddenSize, config.HiddenSize, _spatialMergeSize, name: "merger");

        register_module("patch_embed", _patchEmbed);
        register_module("rotary_pos_emb", _rotaryPosEmb);
        register_module("blocks", _blocks);
        register_module("merger", _merger);
    }

    private Tensor GetRotPosEmb(Tensor gridThw)
    {
        using var scope = torch.NewDisposeScope();
        int numImages = (int)gridThw.shape[0];
        
        var posIdsList = new List<Tensor>();
        for (int i = 0; i < numImages; i++)
        {
            long t = gridThw[i, 0].item<long>();
            long h = gridThw[i, 1].item<long>();
            long w = gridThw[i, 2].item<long>();

            var hposIds = torch.arange(h, device: gridThw.device).unsqueeze(1).expand(h, w);
            hposIds = hposIds.reshape(
                h / _spatialMergeSize, _spatialMergeSize,
                w / _spatialMergeSize, _spatialMergeSize
            );
            hposIds = hposIds.permute(0, 2, 1, 3).flatten();

            var wposIds = torch.arange(w, device: gridThw.device).unsqueeze(0).expand(h, w);
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
        
        return scope.MoveToOuter(rotaryPosEmb.flatten(1));
    }

    private (Tensor windowIndex, Tensor cuWindowSeqlens) GetWindowIndex(Tensor gridThw)
    {
        using var scope = torch.NewDisposeScope();
        
        var windowIndexList = new List<Tensor>();
        var cuWindowSeqlensList = new List<int> { 0 };
        long windowIndexId = 0;
        
        long vitMergerWindowSize = _windowSize / _spatialMergeSize / _patchSize;
        long numImages = gridThw.shape[0];
        
        for (int i = 0; i < numImages; i++)
        {
            long gridT = gridThw[i, 0].item<long>();
            long gridH = gridThw[i, 1].item<long>();
            long gridW = gridThw[i, 2].item<long>();
            
            long llmGridH = gridH / _spatialMergeSize;
            long llmGridW = gridW / _spatialMergeSize;
            
            var index = torch.arange(gridT * llmGridH * llmGridW, device: gridThw.device).reshape(gridT, llmGridH, llmGridW);
            
            long padH = vitMergerWindowSize - (llmGridH % vitMergerWindowSize);
            
            long padW = vitMergerWindowSize - (llmGridW % vitMergerWindowSize);
            
            long numWindowsH = (llmGridH + padH) / vitMergerWindowSize;
            long numWindowsW = (llmGridW + padW) / vitMergerWindowSize;
            
            // F.pad in PyTorch pads last dim backward: left, right, top, bottom.
            var indexPadded = torch.nn.functional.pad(index, new[] { 0L, padW, 0L, padH }, value: -100);
            
            indexPadded = indexPadded.reshape(
                gridT,
                numWindowsH,
                vitMergerWindowSize,
                numWindowsW,
                vitMergerWindowSize
            );
            
            indexPadded = indexPadded.permute(0, 1, 3, 2, 4).reshape(
                gridT,
                numWindowsH * numWindowsW,
                vitMergerWindowSize,
                vitMergerWindowSize
            );
            
            var seqlens = (indexPadded != -100).sum(new long[] { 2, 3 }).reshape(-1);
            indexPadded = indexPadded.reshape(-1);
            
            var indexNew = indexPadded[indexPadded != -100];
            windowIndexList.Add(indexNew + windowIndexId);
            
            // Generate sequence lengths prefix sum
            var seqlensArray = seqlens.data<long>().ToArray();
            int currentCu = cuWindowSeqlensList[^1];
            foreach (var s in seqlensArray)
            {
                currentCu += (int)(s * _spatialMergeUnit);
                cuWindowSeqlensList.Add(currentCu);
            }
            
            windowIndexId += gridT * llmGridH * llmGridW;
        }

        var windowIndex = torch.cat(windowIndexList, dim: 0);
        var uniqueCuWindowSeqlens = new List<long>();
        foreach (var value in cuWindowSeqlensList)
        {
            if (uniqueCuWindowSeqlens.Count == 0 || uniqueCuWindowSeqlens[^1] != value)
            {
                uniqueCuWindowSeqlens.Add(value);
            }
        }

        var cuWindowSeqlensTensor = torch.tensor(uniqueCuWindowSeqlens.ToArray(), dtype: ScalarType.Int32, device: gridThw.device);

        return (scope.MoveToOuter(windowIndex), scope.MoveToOuter(cuWindowSeqlensTensor));
    }

    public Tensor forward(Tensor hiddenStates, Tensor gridThw)
    {
        using var scope = torch.NewDisposeScope();
        
        hiddenStates = _patchEmbed.forward(hiddenStates);
        var rotaryPosEmb = GetRotPosEmb(gridThw);
        
        var (windowIndex, cuWindowSeqlens) = GetWindowIndex(gridThw);
        
        // Unique consecutive
        // Torchsharp might not have unique_consecutive directly, but we can compute it if needed.
        // Actually since we built it monotonic, it is already valid as cu_seqlens.
        
        var seqLen = hiddenStates.shape[0];
        hiddenStates = hiddenStates.reshape(seqLen / _spatialMergeUnit, _spatialMergeUnit, -1);
        hiddenStates = hiddenStates.index_select(0, windowIndex); // hidden_states[window_index, :, :]
        hiddenStates = hiddenStates.reshape(seqLen, -1);
        
        rotaryPosEmb = rotaryPosEmb.reshape(seqLen / _spatialMergeUnit, _spatialMergeUnit, -1);
        rotaryPosEmb = rotaryPosEmb.index_select(0, windowIndex);
        rotaryPosEmb = rotaryPosEmb.reshape(seqLen, -1);
        
        var emb = torch.cat(new[] { rotaryPosEmb, rotaryPosEmb }, dim: -1);
        var cos = emb.cos();
        var sin = emb.sin();

        var patchCounts = gridThw[.., 1] * gridThw[.., 2];
        var patchCountsRep = patchCounts.repeat_interleave(gridThw[.., 0]);
        var cuSeqlens = patchCountsRep.to_type(ScalarType.Int64).cumsum(dim: 0);
        cuSeqlens = torch.nn.functional.pad(cuSeqlens, new long[] { 1, 0 }, value: 0);

        for (int layerNum = 0; layerNum < _blocks.Count; layerNum++)
        {
            var blk = _blocks[layerNum];
            bool isFullAtt = System.Array.IndexOf(_fullAttBlockIndexes, layerNum) >= 0;
            var cuSeqlensNow = isFullAtt ? cuSeqlens : cuWindowSeqlens;
            
            hiddenStates = blk.forward(hiddenStates, cuSeqlensNow, cos, sin);
        }

        var mergedHiddenStates = _merger.forward(hiddenStates);
        
        // Reverse indices sort
        var reverseIndices = torch.argsort(windowIndex);
        mergedHiddenStates = mergedHiddenStates.index_select(0, reverseIndices);

        return scope.MoveToOuter(mergedHiddenStates);
    }
}



