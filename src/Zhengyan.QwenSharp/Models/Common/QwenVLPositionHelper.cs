using System;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.Models.Vision;

public static class QwenVLPositionHelper
{
    public static Tensor BuildPromptPositionIds(
        Tensor inputIds,
        int imageTokenId,
        int spatialMergeSize,
        Tensor? imageGridThw,
        out long ropeDelta)
    {
        using var scope = NewDisposeScope();
        var mmTokenTypeIds = zeros_like(inputIds, dtype: ScalarType.Int64);
        mmTokenTypeIds = where(inputIds == imageTokenId, ones_like(mmTokenTypeIds), mmTokenTypeIds);
        return scope.MoveToOuter(BuildPromptPositionIds(inputIds, mmTokenTypeIds, spatialMergeSize, imageGridThw, out ropeDelta));
    }

    public static Tensor BuildPromptPositionIds(
        Tensor inputIds,
        Tensor mmTokenTypeIds,
        int spatialMergeSize,
        Tensor? imageGridThw,
        out long ropeDelta,
        Tensor? attentionMask = null,
        Tensor? visualTemporalStrides = null)
    {
        using var scope = NewDisposeScope();

        long batchSize = inputIds.shape[0];
        long seqLen = inputIds.shape[1];
        var positionIds = zeros(new long[] { 3, batchSize, seqLen }, dtype: ScalarType.Int64, device: inputIds.device);
        int imageIndex = 0;
        long maxPosition = 0;

        for (int batchIdx = 0; batchIdx < batchSize; batchIdx++)
        {
            using var tokenTypesCpu = mmTokenTypeIds[batchIdx].cpu();
            using var attentionCpu = attentionMask is not null ? attentionMask[batchIdx].cpu() : null;
            var tokenTypes = tokenTypesCpu.data<long>().ToArray();
            var attention = attentionCpu is not null ? attentionCpu.data<long>().ToArray() : null;
            long currentPos = 0;
            int tokenIdx = 0;

            while (tokenIdx < tokenTypes.Length)
            {
                if (attention is not null && attention[tokenIdx] == 0)
                {
                    tokenIdx++;
                    continue;
                }

                long modalityType = tokenTypes[tokenIdx];
                int segmentStart = tokenIdx;
                while (tokenIdx < tokenTypes.Length)
                {
                    if (attention is not null && attention[tokenIdx] == 0)
                    {
                        break;
                    }

                    if (tokenTypes[tokenIdx] != modalityType)
                    {
                        break;
                    }

                    tokenIdx++;
                }

                int segmentLength = tokenIdx - segmentStart;
                if (segmentLength == 0)
                {
                    continue;
                }

                if (modalityType == 0)
                {
                    for (int offset = 0; offset < segmentLength; offset++)
                    {
                        long pos = currentPos + offset;
                        positionIds[0, batchIdx, segmentStart + offset] = pos;
                        positionIds[1, batchIdx, segmentStart + offset] = pos;
                        positionIds[2, batchIdx, segmentStart + offset] = pos;
                        maxPosition = Math.Max(maxPosition, pos);
                    }

                    currentPos += segmentLength;
                    continue;
                }

                if (modalityType != 1 && modalityType != 2)
                {
                    throw new InvalidOperationException($"Unsupported multimodal token type: {modalityType}.");
                }

                int remainingSegmentTokens = segmentLength;
                int writeIdx = segmentStart;
                long segmentBase = currentPos;
                long segmentTemporalOffset = 0;
                long segmentTemporalExtent = 1;
                long segmentMaxH = 1;
                long segmentMaxW = 1;

                while (remainingSegmentTokens > 0)
                {
                    if (imageGridThw is null || imageIndex >= imageGridThw.shape[0])
                    {
                        throw new InvalidOperationException("Image token segment does not match image_grid_thw.");
                    }

                    long gridT = imageGridThw[imageIndex, 0].item<long>();
                    long gridH = imageGridThw[imageIndex, 1].item<long>() / spatialMergeSize;
                    long gridW = imageGridThw[imageIndex, 2].item<long>() / spatialMergeSize;
                    long expectedTokens = gridT * gridH * gridW;
                    if (gridT <= 0 || gridH <= 0 || gridW <= 0)
                    {
                        throw new InvalidOperationException($"Invalid image_grid_thw at index {imageIndex}: t={gridT}, h={gridH}, w={gridW}.");
                    }

                    if (expectedTokens > remainingSegmentTokens)
                    {
                        throw new InvalidOperationException($"Image token count mismatch. tokens={segmentLength}, remaining={remainingSegmentTokens}, expected={expectedTokens}.");
                    }

                    var temporalStride = 1L;
                    if (modalityType == 2 && visualTemporalStrides is not null && imageIndex < visualTemporalStrides.shape[0])
                    {
                        using var strideTensor = visualTemporalStrides[imageIndex].cpu();
                        temporalStride = Math.Max(1L, strideTensor.item<long>());
                    }

                    for (long t = 0; t < gridT; t++)
                    {
                        for (long h = 0; h < gridH; h++)
                        {
                            for (long w = 0; w < gridW; w++)
                            {
                                var temporalPos = segmentBase + ((segmentTemporalOffset + t) * temporalStride);
                                var heightPos = segmentBase + h;
                                var widthPos = segmentBase + w;
                                positionIds[0, batchIdx, writeIdx] = temporalPos;
                                positionIds[1, batchIdx, writeIdx] = heightPos;
                                positionIds[2, batchIdx, writeIdx] = widthPos;
                                maxPosition = Math.Max(maxPosition, Math.Max(temporalPos, Math.Max(heightPos, widthPos)));
                                writeIdx++;
                            }
                        }
                    }

                    remainingSegmentTokens -= (int)expectedTokens;
                    segmentTemporalExtent = Math.Max(segmentTemporalExtent, ((segmentTemporalOffset + gridT - 1) * temporalStride) + 1);
                    segmentTemporalOffset += gridT;
                    segmentMaxH = Math.Max(segmentMaxH, gridH);
                    segmentMaxW = Math.Max(segmentMaxW, gridW);
                    imageIndex++;
                }

                currentPos += Math.Max(segmentTemporalExtent, Math.Max(segmentMaxH, segmentMaxW));
            }
        }

        ropeDelta = maxPosition + 1 - seqLen;
        return scope.MoveToOuter(positionIds);
    }

    public static Tensor BuildDecodePositionIds(Tensor inputIds, KVCache? kvCache, long ropeDelta)
    {
        using var scope = NewDisposeScope();
        long batchSize = inputIds.shape[0];
        long seqLen = inputIds.shape[1];
        long pastLength = kvCache?.GetSeqLength() ?? 0;
        var basePositions = arange(pastLength, pastLength + seqLen, dtype: ScalarType.Int64, device: inputIds.device)
            .view(1, 1, seqLen)
            .expand(3, batchSize, seqLen);
        return scope.MoveToOuter(basePositions + ropeDelta);
    }
}
