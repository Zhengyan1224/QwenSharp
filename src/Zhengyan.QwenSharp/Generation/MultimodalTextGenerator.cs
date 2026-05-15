using System;
using System.Collections.Generic;
using Zhengyan.QwenSharp.Core;
using Zhengyan.QwenSharp.Models;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.Generation;

public static class MultimodalTextGenerator
{
    public static IEnumerable<int> Generate(
        IMultimodalCausalLM model,
        Tensor inputIds,
        Tensor pixelValues,
        Tensor imageGridThw,
        GenerationConfig config,
        params int[] eosTokenIds)
    {
        var kvCache = new KVCache();
        var inputDevice = inputIds.device;
        var currentInputIds = inputIds;
        var ownsCurrentInputIds = false;
        var isFirstStep = true;

        using var noGrad = no_grad();
        const int MaxCacheLen = 2048;

        try
        {
            for (int step = 0; step < config.MaxNewTokens; step++)
            {
                long nextTokenId;

                using (var scope = NewDisposeScope())
                {
                    using var logits = isFirstStep
                        ? model.forward(currentInputIds, kvCache: kvCache, pixelValues: pixelValues, imageGridThw: imageGridThw)
                        : model.forward(currentInputIds, kvCache: kvCache);

                    using var nextTokenLogits = TorchHelper.PrepareSamplingLogits(logits[0, -1, ..]);

                    nextTokenId = TorchHelper.SampleTokenId(nextTokenLogits, config);
                }

                if (Array.IndexOf(eosTokenIds, (int)nextTokenId) >= 0)
                {
                    yield break;
                }

                yield return (int)nextTokenId;

                if (ownsCurrentInputIds)
                {
                    currentInputIds.Dispose();
                }

                using (var nextTokenScope = NewDisposeScope())
                {
                    var nextTokenTensor = tensor(new long[] { nextTokenId }, dtype: ScalarType.Int64, device: inputDevice);
                    currentInputIds = nextTokenScope.MoveToOuter(nextTokenTensor.unsqueeze(0));
                }
                ownsCurrentInputIds = true;
                isFirstStep = false;

                if (kvCache.GetSeqLength() > MaxCacheLen)
                {
                    kvCache.Trim(MaxCacheLen);
                }
            }
        }
        finally
        {
            if (ownsCurrentInputIds)
            {
                currentInputIds.Dispose();
            }

            kvCache.Dispose();
        }
    }

}
