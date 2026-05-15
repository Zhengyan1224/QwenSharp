using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Common;

/// <summary>
/// Grouped Query Attention (GQA) with RoPE, KV caching, and standard SDPA attention.
/// </summary>
public class Attention : nn.Module
{
    private readonly long _hiddenSize;
    private readonly long _numHeads;
    private readonly long _numKvHeads;
    private readonly long _headDim;
    private readonly long _numKeyValueGroups;
    private readonly long _attnOutputSize;

    private readonly Linear _qProj;
    private readonly Linear _kProj;
    private readonly Linear _vProj;
    private readonly Linear _oProj;
    private readonly RMSNorm? _qNorm;
    private readonly RMSNorm? _kNorm;

    /// <summary>
    /// Retained for compatibility. The cross-platform build always uses SDPA.
    /// </summary>
    public static bool UseFlashAttention { get; set; } = false;

    public Attention(
        long hiddenSize,
        long numHeads,
        long numKvHeads,
        long headDim = -1,
        bool qkvBias = true,
        bool oProjBias = false,
        bool useQkNorm = false,
        double rmsNormEps = 1e-6,
        string name = "Attention") : base(name)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim > 0 ? headDim : hiddenSize / numHeads;
        _numKeyValueGroups = numHeads / numKvHeads;
        _attnOutputSize = _numHeads * _headDim;

        _qProj = Linear(hiddenSize, numHeads * _headDim, hasBias: qkvBias);
        _kProj = Linear(hiddenSize, numKvHeads * _headDim, hasBias: qkvBias);
        _vProj = Linear(hiddenSize, numKvHeads * _headDim, hasBias: qkvBias);
        _oProj = Linear(numHeads * _headDim, hiddenSize, hasBias: oProjBias);

        if (useQkNorm)
        {
            _qNorm = new RMSNorm(_headDim, rmsNormEps, name: "q_norm");
            _kNorm = new RMSNorm(_headDim, rmsNormEps, name: "k_norm");
            register_module("q_norm", _qNorm);
            register_module("k_norm", _kNorm);
        }

        register_module("q_proj", _qProj);
        register_module("k_proj", _kProj);
        register_module("v_proj", _vProj);
        register_module("o_proj", _oProj);
    }

    public Tensor forward(
        Tensor hiddenStates,
        Tensor positionIds,
        Tensor cos,
        Tensor sin,
        Tensor? attentionMask = null,
        KVCache? kvCache = null,
        int layerIdx = -1)
    {
        using var scope = torch.NewDisposeScope();

        long bsz = hiddenStates.shape[0];
        long qLen = hiddenStates.shape[1];

        var queryStates = _qProj.forward(hiddenStates).view(bsz, qLen, _numHeads, _headDim);
        var keyStates = _kProj.forward(hiddenStates).view(bsz, qLen, _numKvHeads, _headDim);
        var valueStates = _vProj.forward(hiddenStates).view(bsz, qLen, _numKvHeads, _headDim);

        if (_qNorm != null) queryStates = _qNorm.forward(queryStates);
        if (_kNorm != null) keyStates = _kNorm.forward(keyStates);

        var queryT = queryStates.transpose(1, 2);
        var keyT = keyStates.transpose(1, 2);
        var valueT = valueStates.transpose(1, 2);

        (queryT, keyT) = RotaryEmbedding.ApplyRotaryPosEmb(queryT, keyT, cos, sin);

        if (kvCache != null)
        {
            scope.Detach(keyT);
            scope.Detach(valueT);
            (keyT, valueT) = kvCache.Update(keyT, valueT, layerIdx);
            scope.Detach(keyT);
            scope.Detach(valueT);
        }

        keyT = RepeatKV(keyT, _numKeyValueGroups);
        valueT = RepeatKV(valueT, _numKeyValueGroups);

        bool isCausal = attentionMask is null && qLen > 1;
        var attnOutput = torch.nn.functional.scaled_dot_product_attention(
            queryT,
            keyT,
            valueT,
            attentionMask,
            0.0,
            isCausal);

        attnOutput = attnOutput.transpose(1, 2).contiguous().view(bsz, qLen, _attnOutputSize);
        var output = _oProj.forward(attnOutput);
        return scope.MoveToOuter(output);
    }

    private static Tensor RepeatKV(Tensor hiddenStates, long nRep)
    {
        if (nRep == 1)
        {
            return hiddenStates;
        }

        long bsz = hiddenStates.shape[0];
        long numKvHeads = hiddenStates.shape[1];
        long seqLen = hiddenStates.shape[2];
        long headDim = hiddenStates.shape[3];

        var expanded = hiddenStates.unsqueeze(2).expand(bsz, numKvHeads, nRep, seqLen, headDim);
        return expanded.reshape(bsz, numKvHeads * nRep, seqLen, headDim);
    }
}
