using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen2Audio;

internal sealed class SinusoidsPositionEmbedding : nn.Module
{
    private readonly Tensor _positionalEmbedding;

    public SinusoidsPositionEmbedding(long length, long channels, string name = "positional_embedding") : base(name)
    {
        if ((channels & 1) != 0)
        {
            throw new ArgumentException("Sinusoidal position embedding requires an even channel count.", nameof(channels));
        }

        var half = channels / 2;
        var values = new float[length * channels];
        var logTimescaleIncrement = Math.Log(10000.0) / (half - 1);

        for (var pos = 0; pos < length; pos++)
        {
            var rowOffset = pos * channels;
            for (var i = 0; i < half; i++)
            {
                var scaledTime = pos * Math.Exp(-logTimescaleIncrement * i);
                values[rowOffset + i] = (float)Math.Sin(scaledTime);
                values[rowOffset + half + i] = (float)Math.Cos(scaledTime);
            }
        }

        _positionalEmbedding = torch.tensor(values, dtype: ScalarType.Float32).view(length, channels);
        register_buffer("positional_embedding", _positionalEmbedding, persistent: false);
    }

    public Tensor forward(long seqLen, ScalarType dtype, Device device)
    {
        var embedding = _positionalEmbedding[..(int)seqLen, ..];
        return embedding.dtype != dtype || embedding.device != device
            ? embedding.to(dtype, device)
            : embedding;
    }
}

public class Qwen2AudioAttention : nn.Module
{
    private readonly int _embedDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly Linear _qProj;
    private readonly Linear _kProj;
    private readonly Linear _vProj;
    private readonly Linear _outProj;

    public Qwen2AudioAttention(Qwen2AudioEncoderConfig config, string name = "self_attn") : base(name)
    {
        _embedDim = config.DModel;
        _numHeads = config.EncoderAttentionHeads;
        _headDim = _embedDim / _numHeads;

        _qProj = Linear(_embedDim, _embedDim, hasBias: true);
        _kProj = Linear(_embedDim, _embedDim, hasBias: false);
        _vProj = Linear(_embedDim, _embedDim, hasBias: true);
        _outProj = Linear(_embedDim, _embedDim, hasBias: true);

        register_module("q_proj", _qProj);
        register_module("k_proj", _kProj);
        register_module("v_proj", _vProj);
        register_module("out_proj", _outProj);
    }

    public Tensor forward(Tensor hiddenStates, Tensor? attentionMask = null)
    {
        using var scope = torch.NewDisposeScope();
        
        var bsz = hiddenStates.shape[0];
        var tgtLen = hiddenStates.shape[1];

        // TorchSharp nn.Linear automatically broadcasts over batch and sequence lengths
        var queryStates = _qProj.forward(hiddenStates);
        var keyStates = _kProj.forward(hiddenStates);
        var valueStates = _vProj.forward(hiddenStates);

        queryStates = queryStates.view(bsz, tgtLen, _numHeads, _headDim).transpose(1, 2);
        keyStates = keyStates.view(bsz, tgtLen, _numHeads, _headDim).transpose(1, 2);
        valueStates = valueStates.view(bsz, tgtLen, _numHeads, _headDim).transpose(1, 2);

        var attnOutput = torch.nn.functional.scaled_dot_product_attention(
            queryStates, keyStates, valueStates,
            attentionMask,
            0.0
        );

        attnOutput = attnOutput.transpose(1, 2).contiguous().view(bsz, tgtLen, _embedDim);
        var output = _outProj.forward(attnOutput);

        return scope.MoveToOuter(output);
    }

    public Tensor forwardPacked(Tensor hiddenStates, Tensor attentionMask)
    {
        using var scope = torch.NewDisposeScope();

        var seqLen = hiddenStates.shape[0];

        var queryStates = _qProj.forward(hiddenStates).view(seqLen, _numHeads, _headDim).transpose(0, 1);
        var keyStates = _kProj.forward(hiddenStates).view(seqLen, _numHeads, _headDim).transpose(0, 1);
        var valueStates = _vProj.forward(hiddenStates).view(seqLen, _numHeads, _headDim).transpose(0, 1);

        var attnOutput = torch.nn.functional.scaled_dot_product_attention(
            queryStates,
            keyStates,
            valueStates,
            attentionMask,
            0.0,
            false);

        attnOutput = attnOutput.transpose(0, 1).contiguous().view(seqLen, _embedDim);
        var output = _outProj.forward(attnOutput);

        return scope.MoveToOuter(output);
    }
}

public class Qwen2AudioEncoderLayer : nn.Module
{
    private readonly Qwen2AudioAttention _selfAttn;
    private readonly LayerNorm _selfAttnLayerNorm;
    private readonly Linear _fc1;
    private readonly Linear _fc2;
    private readonly LayerNorm _finalLayerNorm;

    public Qwen2AudioEncoderLayer(Qwen2AudioEncoderConfig config, string name = "layer") : base(name)
    {
        _selfAttn = new Qwen2AudioAttention(config);
        _selfAttnLayerNorm = LayerNorm(new long[] { config.DModel });
        _fc1 = Linear(config.DModel, config.EncoderFfnDim);
        _fc2 = Linear(config.EncoderFfnDim, config.DModel);
        _finalLayerNorm = LayerNorm(new long[] { config.DModel });

        register_module("self_attn", _selfAttn);
        register_module("self_attn_layer_norm", _selfAttnLayerNorm);
        register_module("fc1", _fc1);
        register_module("fc2", _fc2);
        register_module("final_layer_norm", _finalLayerNorm);
    }

    public Tensor forward(Tensor hiddenStates, Tensor? attentionMask = null)
    {
        using var scope = torch.NewDisposeScope();

        var residual = hiddenStates;
        hiddenStates = _selfAttnLayerNorm.forward(hiddenStates);
        hiddenStates = _selfAttn.forward(hiddenStates, attentionMask);
        hiddenStates = residual + hiddenStates;

        residual = hiddenStates;
        hiddenStates = _finalLayerNorm.forward(hiddenStates);
        hiddenStates = torch.nn.functional.gelu(_fc1.forward(hiddenStates));
        hiddenStates = _fc2.forward(hiddenStates);
        hiddenStates = residual + hiddenStates;

        return scope.MoveToOuter(hiddenStates);
    }

    public Tensor forwardPacked(Tensor hiddenStates, Tensor attentionMask)
    {
        using var scope = torch.NewDisposeScope();

        var residual = hiddenStates;
        hiddenStates = _selfAttnLayerNorm.forward(hiddenStates);
        hiddenStates = _selfAttn.forwardPacked(hiddenStates, attentionMask);
        hiddenStates = residual + hiddenStates;

        residual = hiddenStates;
        hiddenStates = _finalLayerNorm.forward(hiddenStates);
        hiddenStates = torch.nn.functional.gelu(_fc1.forward(hiddenStates));
        hiddenStates = _fc2.forward(hiddenStates);
        hiddenStates = residual + hiddenStates;

        return scope.MoveToOuter(hiddenStates);
    }
}

public class Qwen2AudioEncoder : nn.Module
{
    private readonly Qwen2AudioEncoderConfig _config;
    private readonly Conv1d _conv1;
    private readonly Conv1d _conv2;
    private readonly Embedding? _embedPositions;
    private readonly SinusoidsPositionEmbedding? _positionalEmbedding;
    private readonly ModuleList<Qwen2AudioEncoderLayer> _layers;
    private readonly LayerNorm _layerNorm;
    private readonly AvgPool1d _avgPooler;
    private readonly Linear? _proj;

    public Qwen2AudioEncoder(Qwen2AudioEncoderConfig config, string name = "model") : base(name)
    {
        _config = config;
        _conv1 = Conv1d(config.NumMelBins, config.DModel, kernel_size: 3, padding: 1);
        _conv2 = Conv1d(config.DModel, config.DModel, kernel_size: 3, stride: 2, padding: 1);

        if (config.UseSinusoidalPositionEmbedding)
        {
            _positionalEmbedding = new SinusoidsPositionEmbedding(config.MaxSourcePositions, config.DModel);
        }
        else
        {
            _embedPositions = Embedding(config.MaxSourcePositions, config.DModel);
            foreach (var p in _embedPositions.parameters()) p.requires_grad = false;
        }

        _layers = new ModuleList<Qwen2AudioEncoderLayer>();
        for (int i = 0; i < config.EncoderLayers; i++)
        {
            _layers.append(new Qwen2AudioEncoderLayer(config, name: $"layers.{i}"));
        }

        _layerNorm = LayerNorm(new long[] { config.DModel });
        _avgPooler = AvgPool1d(kernel_size: 2L, stride: 2L);
        if (config.ProjectOutput)
        {
            var outputDim = config.OutputDim > 0 ? config.OutputDim : config.DModel;
            _proj = Linear(config.DModel, outputDim, hasBias: true);
        }

        register_module("conv1", _conv1);
        register_module("conv2", _conv2);
        if (_positionalEmbedding is not null)
        {
            register_module("positional_embedding", _positionalEmbedding);
        }
        else
        {
            register_module("embed_positions", _embedPositions!);
        }

        register_module("layers", _layers);
        register_module(config.UseOmniAudioChunking ? "ln_post" : "layer_norm", _layerNorm);
        register_module("avg_pooler", _avgPooler);
        if (_proj is not null)
        {
            register_module("proj", _proj);
        }
    }

    public Tensor forward(Tensor inputFeatures, Tensor? attentionMask = null)
    {
        using var scope = torch.NewDisposeScope();

        var convDtype = _conv1.weight!.dtype;
        var convDevice = _conv1.weight!.device;
        if (inputFeatures.dtype != convDtype || inputFeatures.device != convDevice)
        {
            inputFeatures = inputFeatures.to(convDtype, convDevice);
        }

        if (_config.UseOmniAudioChunking)
        {
            return scope.MoveToOuter(ForwardOmniChunked(inputFeatures));
        }

        // inputFeatures: [batch_size, num_mel_bins, seq_length]
        var hiddenStates = torch.nn.functional.gelu(_conv1.forward(inputFeatures));
        hiddenStates = torch.nn.functional.gelu(_conv2.forward(hiddenStates));

        // After convs: [batch_size, d_model, new_seq_length]
        hiddenStates = hiddenStates.permute(0, 2, 1); // [batch_size, new_seq_length, d_model]

        var seqLen = hiddenStates.shape[1];
        var embedPos = GetPositionEmbedding(seqLen, hiddenStates.dtype, hiddenStates.device);
        hiddenStates = hiddenStates + embedPos.unsqueeze(0);

        foreach (var layer in _layers)
        {
            hiddenStates = layer.forward(hiddenStates, attentionMask); // [batch_size, new_seq_length, d_model]
        }

        hiddenStates = hiddenStates.permute(0, 2, 1); // [batch_size, d_model, new_seq_length]
        hiddenStates = _avgPooler.forward(hiddenStates);
        hiddenStates = hiddenStates.permute(0, 2, 1); // [batch_size, pooled_seq_length, d_model]

        hiddenStates = _layerNorm.forward(hiddenStates);

        return scope.MoveToOuter(hiddenStates);
    }

    private Tensor ForwardOmniChunked(Tensor inputFeatures)
    {
        using var scope = torch.NewDisposeScope();

        if (inputFeatures.ndim == 2)
        {
            inputFeatures = inputFeatures.unsqueeze(0);
        }

        if (inputFeatures.shape[1] != _config.NumMelBins && inputFeatures.shape[2] == _config.NumMelBins)
        {
            inputFeatures = inputFeatures.permute(0, 2, 1);
        }

        var batchSize = (int)inputFeatures.shape[0];
        var numMelBins = (int)inputFeatures.shape[1];
        var featureLen = (int)inputFeatures.shape[2];
        var chunkFrameLen = _config.NWindow > 0 ? _config.NWindow * 2 : _config.MaxSourcePositions * 2;
        var chunkLengths = new List<int>();
        var audioAfterCnnLengths = new List<int>(batchSize);

        for (var batchIdx = 0; batchIdx < batchSize; batchIdx++)
        {
            var audioAfterCnnLen = 0;
            for (var start = 0; start < featureLen; start += chunkFrameLen)
            {
                var length = Math.Min(chunkFrameLen, featureLen - start);
                chunkLengths.Add(length);
                audioAfterCnnLen += GetConvStride2OutputLength(length);
            }

            audioAfterCnnLengths.Add(audioAfterCnnLen);
        }

        if (chunkLengths.Count == 0)
        {
            chunkLengths.Add(1);
            audioAfterCnnLengths.Add(1);
        }

        var maxChunkLen = 0;
        foreach (var length in chunkLengths)
        {
            maxChunkLen = Math.Max(maxChunkLen, length);
        }

        var paddedFeature = torch.zeros(
            new long[] { chunkLengths.Count, numMelBins, maxChunkLen },
            dtype: inputFeatures.dtype,
            device: inputFeatures.device);
        var paddedMask = torch.zeros(
            new long[] { chunkLengths.Count, 1, maxChunkLen },
            dtype: inputFeatures.dtype,
            device: inputFeatures.device);

        var chunkIdx = 0;
        for (var batchIdx = 0; batchIdx < batchSize; batchIdx++)
        {
            for (var start = 0; start < featureLen; start += chunkFrameLen)
            {
                var length = Math.Min(chunkFrameLen, featureLen - start);
                using var chunk = inputFeatures[batchIdx, .., start..(start + length)];
                paddedFeature[chunkIdx, .., ..length] = chunk;
                paddedMask[chunkIdx, .., ..length] = 1;
                chunkIdx++;
            }
        }

        var paddedEmbed = torch.nn.functional.gelu(_conv1.forward(paddedFeature)) * paddedMask;
        paddedEmbed = torch.nn.functional.gelu(_conv2.forward(paddedEmbed)).transpose(1, 2);

        var positionEmbedding = GetPositionEmbedding(paddedEmbed.shape[1], paddedEmbed.dtype, paddedEmbed.device);
        paddedEmbed = paddedEmbed + positionEmbedding.unsqueeze(0);

        var hiddenParts = new List<Tensor>(chunkLengths.Count);
        var cuSeqlens = new List<long>(chunkLengths.Count + 1) { 0 };
        long runningSeqLen = 0;
        for (var i = 0; i < chunkLengths.Count; i++)
        {
            var afterCnnLen = GetConvStride2OutputLength(chunkLengths[i]);
            hiddenParts.Add(paddedEmbed[i, ..afterCnnLen, ..]);
            runningSeqLen += afterCnnLen;
            cuSeqlens.Add(runningSeqLen);
        }

        var hiddenStates = torch.cat(hiddenParts.ToArray(), dim: 0);
        var packedAttentionMask = CreatePackedAttentionMask(cuSeqlens, hiddenStates.dtype, hiddenStates.device);

        foreach (var layer in _layers)
        {
            hiddenStates = layer.forwardPacked(hiddenStates, packedAttentionMask);
        }

        var audioParts = new List<Tensor>(audioAfterCnnLengths.Count);
        var offset = 0;
        foreach (var audioAfterCnnLen in audioAfterCnnLengths)
        {
            var audioStates = hiddenStates[offset..(offset + audioAfterCnnLen), ..];
            offset += audioAfterCnnLen;

            Tensor tokenAudio;
            if (audioAfterCnnLen < 2)
            {
                tokenAudio = audioStates;
            }
            else
            {
                tokenAudio = _avgPooler.forward(audioStates.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1);
            }

            tokenAudio = _layerNorm.forward(tokenAudio);
            if (_proj is not null)
            {
                tokenAudio = _proj.forward(tokenAudio);
            }

            audioParts.Add(tokenAudio);
        }

        var output = torch.cat(audioParts.ToArray(), dim: 0);
        return scope.MoveToOuter(output);
    }

    private Tensor GetPositionEmbedding(long seqLen, ScalarType dtype, Device device)
    {
        if (seqLen > _config.MaxSourcePositions)
        {
            throw new InvalidOperationException(
                $"Audio sequence length after convolution ({seqLen}) exceeds max_source_positions ({_config.MaxSourcePositions}). " +
                "Qwen2.5-Omni audio must use the chunked encoder path.");
        }

        if (_positionalEmbedding is not null)
        {
            return _positionalEmbedding.forward(seqLen, dtype, device);
        }

        var embedPos = _embedPositions!.weight![..(int)seqLen, ..];
        return embedPos.dtype != dtype || embedPos.device != device
            ? embedPos.to(dtype, device)
            : embedPos;
    }

    private static Tensor CreatePackedAttentionMask(IReadOnlyList<long> cuSeqlens, ScalarType dtype, Device device)
    {
        var totalSeqLen = cuSeqlens[^1];
        var attentionMask = torch.ones(
            new long[] { 1, totalSeqLen, totalSeqLen },
            dtype: dtype,
            device: device) * -10000.0;

        for (var i = 1; i < cuSeqlens.Count; i++)
        {
            var start = (int)cuSeqlens[i - 1];
            var end = (int)cuSeqlens[i];
            attentionMask[0, start..end, start..end] = 0;
        }

        return attentionMask;
    }

    private static int GetConvStride2OutputLength(int inputLength)
        => Math.Max(1, (inputLength - 1) / 2 + 1);
}
