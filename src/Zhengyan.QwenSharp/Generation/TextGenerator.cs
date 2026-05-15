using System;
using System.Collections.Generic;
using TorchSharp;
using Zhengyan.QwenSharp.Core;
using Zhengyan.QwenSharp.Models;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.Generation;

/// <summary>
/// Implementation of text generation strategies including Greedy Search and Top-P/Top-K Nucleus Sampling.
/// </summary>
public static class TextGenerator
{
    /// <summary>
    /// Generates tokens autoregressively from the input prompt.
    /// </summary>
    public static IEnumerable<int> Generate(
        ICausalLM model,
        Tensor inputIds,
        GenerationConfig config,
        params int[] eosTokenIds)
    {
        var kvCache = new KVCache();
        var inputDevice = inputIds.device;
        var currentInputIds = inputIds; // [1, seq_len]
        var ownsCurrentInputIds = false;
        
        using var no_grad = torch.no_grad(); // Disable autograd to prevent OOM
        const int MaxCacheLen = 2048; // sliding window to prevent CUDA OOM
        try
        {
            for (int step = 0; step < config.MaxNewTokens; step++)
            {
                long nextTokenId;
                using (var scope = torch.NewDisposeScope())
                {
                    using var logits = model.forward(currentInputIds, kvCache: kvCache);
                    using var nextTokenLogits = TorchHelper.PrepareSamplingLogits(logits[0, -1, ..]); // [vocab_size]

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

                using (var nextTokenScope = torch.NewDisposeScope())
                {
                    var nextTokenTensor = torch.tensor(new long[] { nextTokenId }, dtype: ScalarType.Int64, device: inputDevice);
                    currentInputIds = nextTokenScope.MoveToOuter(nextTokenTensor.unsqueeze(0));
                }
                ownsCurrentInputIds = true;

                // Trim KV cache to prevent unbounded VRAM growth
                if (kvCache.GetSeqLength() > MaxCacheLen)
                {
                    kvCache.Trim(MaxCacheLen);
                }

                // Force garbage collector to run periodically to reclaim TorchSharp managed wrappers.
                // GC.Collect + WaitForPendingFinalizers triggers TorchSharp tensor finalizers,
                // which release the underlying CUDA memory (equivalent to torch.cuda.empty_cache in Python).
                if ((step + 1) % 30 == 0)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
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
