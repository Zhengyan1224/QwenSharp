using System;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen35;

public static class GatedDeltaNetHelper
{
    public static Tensor L2Norm(Tensor x, int dim = -1, double eps = 1e-6)
    {
        var invNorm = torch.rsqrt((x * x).sum(new long[] { dim }, keepdim: true) + eps);
        return x * invNorm;
    }

    public static Tensor TorchCausalConv1dUpdate(Tensor hiddenStates, Tensor convState, Tensor weight, Tensor? bias = null)
    {
        using var scope = torch.NewDisposeScope();
        var hiddenSize = hiddenStates.shape[1];
        var seqLen = hiddenStates.shape[2];
        var stateLen = convState.shape[2];

        var hiddenStatesNew = torch.cat(new[] { convState, hiddenStates }, dim: -1).to(weight.dtype);
        convState.copy_(hiddenStatesNew[.., .., ^(int)stateLen..]);
        var out_ = torch.nn.functional.conv1d(hiddenStatesNew, weight.unsqueeze(1), bias, padding: 0, groups: hiddenSize);
        out_ = torch.nn.functional.silu(out_[.., .., ^(int)seqLen..]).to(hiddenStates.dtype);
        return scope.MoveToOuter(out_);
    }

    // Simplified fallback to recurrent delta rule for all sequence lengths
    // It is slightly slower but avoids the huge complexity of chunk rule
    public static (Tensor coreAttnOut, Tensor? lastRecurrentState) TorchRecurrentGatedDeltaRule(
        Tensor query, Tensor key, Tensor value, Tensor g, Tensor beta, Tensor? initialState, bool outputFinalState, bool useQkL2NormInKernel)
    {
        using var scope = torch.NewDisposeScope();
        var initialDtype = query.dtype;

        if (useQkL2NormInKernel)
        {
            query = L2Norm(query, dim: -1, eps: 1e-6);
            key = L2Norm(key, dim: -1, eps: 1e-6);
        }

        query = query.transpose(1, 2).contiguous().to(ScalarType.Float32);
        key = key.transpose(1, 2).contiguous().to(ScalarType.Float32);
        value = value.transpose(1, 2).contiguous().to(ScalarType.Float32);
        beta = beta.transpose(1, 2).contiguous().to(ScalarType.Float32);
        g = g.transpose(1, 2).contiguous().to(ScalarType.Float32);

        var batchSize = key.shape[0];
        var numHeads = key.shape[1];
        var sequenceLength = key.shape[2];
        var kHeadDim = key.shape[3];
        var vHeadDim = value.shape[3];

        var scale = 1.0 / System.Math.Sqrt(query.shape[3]);
        query = query * scale;

        var coreAttnOut = torch.zeros(new long[] { batchSize, numHeads, sequenceLength, vHeadDim }, dtype: value.dtype, device: value.device);
        var lastRecurrentState = initialState is null 
            ? torch.zeros(new long[] { batchSize, numHeads, kHeadDim, vHeadDim }, dtype: value.dtype, device: value.device) 
            : initialState.to(value.dtype).to(value.device);

        for (int i = 0; i < sequenceLength; i++)
        {
            var qT = query[.., .., i];
            var kT = key[.., .., i];
            var vT = value[.., .., i];
            var gT = g[.., .., i].exp().unsqueeze(-1).unsqueeze(-1);
            var betaT = beta[.., .., i].unsqueeze(-1);

            lastRecurrentState = lastRecurrentState * gT;
            var kvMem = (lastRecurrentState * kT.unsqueeze(-1)).sum(new long[] { -2 });
            var delta = (vT - kvMem) * betaT;
            lastRecurrentState = lastRecurrentState + kT.unsqueeze(-1) * delta.unsqueeze(-2);
            var out_i = (lastRecurrentState * qT.unsqueeze(-1)).sum(new long[] { -2 });
            coreAttnOut[.., .., i] = out_i;
        }

        if (!outputFinalState)
        {
            lastRecurrentState?.Dispose();
            lastRecurrentState = null;
        }

        coreAttnOut = coreAttnOut.transpose(1, 2).contiguous().to(initialDtype);
        return (scope.MoveToOuter(coreAttnOut), lastRecurrentState is null ? null : scope.MoveToOuter(lastRecurrentState));
    }

    private static ScalarType StringToType(Tensor t)
    {
        return t.dtype;
    }
}
