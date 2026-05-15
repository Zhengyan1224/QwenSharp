using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen35;

public class Qwen35Attention : nn.Module
{
    private readonly long _hiddenSize;
    private readonly long _numHeads;
    private readonly long _numKvHeads;
    private readonly long _headDim;
    private readonly long _numKeyValueGroups;
    private readonly double _scaling;

    private readonly Linear _qProj;
    private readonly Linear _kProj;
    private readonly Linear _vProj;
    private readonly Linear _oProj;
    
    private readonly Qwen35RMSNorm _qNorm;
    private readonly Qwen35RMSNorm _kNorm;

    public Qwen35Attention(Qwen35Config config, int layerIdx, string name = "self_attn") : base(name)
    {
        _hiddenSize = config.HiddenSize;
        _numHeads = config.NumAttentionHeads;
        _numKvHeads = config.NumKeyValueHeads;
        _headDim = config.HeadDim;
        _numKeyValueGroups = _numHeads / _numKvHeads;
        _scaling = System.Math.Pow(_headDim, -0.5);

        // Qwen3.5: q_proj goes to head_dim * 2 (one for query, one for gate)
        _qProj = Linear(_hiddenSize, _numHeads * _headDim * 2, hasBias: config.AttentionBias);
        _kProj = Linear(_hiddenSize, _numKvHeads * _headDim, hasBias: config.AttentionBias);
        _vProj = Linear(_hiddenSize, _numKvHeads * _headDim, hasBias: config.AttentionBias);
        _oProj = Linear(_numHeads * _headDim, _hiddenSize, hasBias: config.AttentionBias);

        _qNorm = new Qwen35RMSNorm(_headDim, config.RmsNormEps, name: "q_norm");
        _kNorm = new Qwen35RMSNorm(_headDim, config.RmsNormEps, name: "k_norm");

        register_module("q_proj", _qProj);
        register_module("k_proj", _kProj);
        register_module("v_proj", _vProj);
        register_module("o_proj", _oProj);
        register_module("q_norm", _qNorm);
        register_module("k_norm", _kNorm);
    }

    public Tensor forward(Tensor hiddenStates, Tensor positionIds, Tensor cos, Tensor sin, Tensor? attentionMask = null, KVCache? kvCache = null, int layerIdx = -1)
    {
        using var scope = torch.NewDisposeScope();

        var bsz = hiddenStates.shape[0];
        var qLen = hiddenStates.shape[1];

        // 1. Q projection -> chunk
        var qProjOutput = _qProj.forward(hiddenStates).view(bsz, qLen, _numHeads, _headDim * 2);
        var chunks = qProjOutput.chunk(2, dim: -1);
        var queryStates = chunks[0];
        var gate = chunks[1].reshape(bsz, qLen, -1); // [bsz, qLen, num_heads * head_dim]

        queryStates = _qNorm.forward(queryStates).transpose(1, 2);

        // 2. K, V projection
        var keyStates = _kProj.forward(hiddenStates).view(bsz, qLen, _numKvHeads, _headDim);
        keyStates = _kNorm.forward(keyStates).transpose(1, 2);
        
        var valueStates = _vProj.forward(hiddenStates).view(bsz, qLen, _numKvHeads, _headDim).transpose(1, 2);

        // 3. Apply RoPE
        (queryStates, keyStates) = Qwen35RotaryEmbedding.ApplyRotaryPosEmb(queryStates, keyStates, cos, sin);

        // 4. KV Cache update
        if (kvCache != null)
        {
            scope.Detach(keyStates);
            scope.Detach(valueStates);
            (keyStates, valueStates) = kvCache.Update(keyStates, valueStates, layerIdx);
            scope.Detach(keyStates);
            scope.Detach(valueStates);
        }

        // 5. Repeat KV for GQA
        keyStates = RepeatKV(keyStates, _numKeyValueGroups);
        valueStates = RepeatKV(valueStates, _numKeyValueGroups);

        // 6. SDPA (scaled dot product attention)
        bool isCausal = attentionMask is null && qLen > 1;
        var attnOutput = torch.nn.functional.scaled_dot_product_attention(
            queryStates, keyStates, valueStates,
            attentionMask,
            0.0,
            isCausal
        );

        // 7. Apply gate and output projection
        attnOutput = attnOutput.transpose(1, 2).contiguous().view(bsz, qLen, -1);
        attnOutput = attnOutput * torch.sigmoid(gate);
        
        var output = _oProj.forward(attnOutput);

        return scope.MoveToOuter(output);
    }

    private Tensor RepeatKV(Tensor hiddenStates, long nRep)
    {
        if (nRep == 1) return hiddenStates;

        var bsz = hiddenStates.shape[0];
        var numKvHeads = hiddenStates.shape[1];
        var seqLen = hiddenStates.shape[2];
        var headDim = hiddenStates.shape[3];

        var expanded = hiddenStates.unsqueeze(2).expand(bsz, numKvHeads, nRep, seqLen, headDim);
        return expanded.reshape(bsz, numKvHeads * nRep, seqLen, headDim);
    }
}
