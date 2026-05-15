using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.Models.Common;

/// <summary>
/// A dynamic cache for holding key/value tensors during autoregressive generation.
/// Equivalent to HuggingFace DynamicCache.
/// </summary>
public class KVCache
{
    private readonly List<Tensor> _keyCache = new();
    private readonly List<Tensor> _valueCache = new();
    private readonly List<Tensor> _convStates = new();
    private readonly List<Tensor> _recurrentStates = new();

    public int SeenTokens { get; private set; }

    /// <summary>
    /// Updates the cache with new key and value states, and returns the concatenated states.
    /// Caller must ensure that passed tensors are detached/moved so they aren't prematurely disposed.
    /// </summary>
    public (Tensor key, Tensor value) Update(Tensor keyStates, Tensor valueStates, int layerIdx)
    {
        // keyStates: [bsz, num_kv_heads, seq_len, head_dim]
        
        if (_keyCache.Count <= layerIdx)
        {
            // Pad cache list
            while (_keyCache.Count <= layerIdx)
            {
                _keyCache.Add(null!);
                _valueCache.Add(null!);
            }
        }

        if (_keyCache[layerIdx] is null)
        {
            _keyCache[layerIdx] = keyStates;
            _valueCache[layerIdx] = valueStates;
            if (layerIdx == 0) SeenTokens = (int)keyStates.shape[2];
        }
        else
        {
            var oldKey = _keyCache[layerIdx];
            var oldValue = _valueCache[layerIdx];
            
            _keyCache[layerIdx] = torch.cat(new[] { oldKey, keyStates }, dim: 2);
            _valueCache[layerIdx] = torch.cat(new[] { oldValue, valueStates }, dim: 2);
            
            // Free the old cached tensors safely
            oldKey.Dispose();
            oldValue.Dispose();
            
            keyStates.Dispose();
            valueStates.Dispose();
            
            if (layerIdx == 0) SeenTokens = (int)_keyCache[0].shape[2];
        }

        // Return the updated full tensors
        return (_keyCache[layerIdx], _valueCache[layerIdx]);
    }

    private void EnsureLinearCapacity(int layerIdx)
    {
        while (_convStates.Count <= layerIdx)
        {
            _convStates.Add(null!);
            _recurrentStates.Add(null!);
        }
    }

    public bool HasLinearState(int layerIdx)
    {
        return layerIdx < _convStates.Count && _convStates[layerIdx] is not null;
    }

    public Tensor? GetConvState(int layerIdx)
    {
        if (layerIdx >= _convStates.Count) return null;
        return _convStates[layerIdx];
    }

    public Tensor? GetRecurrentState(int layerIdx)
    {
        if (layerIdx >= _recurrentStates.Count) return null;
        return _recurrentStates[layerIdx];
    }

    public void SetConvState(int layerIdx, Tensor convState)
    {
        EnsureLinearCapacity(layerIdx);
        _convStates[layerIdx]?.Dispose();
        _convStates[layerIdx] = convState;
    }

    public void SetRecurrentState(int layerIdx, Tensor? recurrentState)
    {
        EnsureLinearCapacity(layerIdx);
        _recurrentStates[layerIdx]?.Dispose();
        _recurrentStates[layerIdx] = recurrentState!;
    }

    public long GetSeqLength(int layerIdx = 0)
    {
        if (_keyCache.Count <= layerIdx || _keyCache[layerIdx] is null) return 0;
        return _keyCache[layerIdx].shape[2];
    }

    /// <summary>
    /// Trims the KV cache to retain only the last <paramref name="maxLen"/> tokens.
    /// Call periodically to free VRAM during long generation.
    /// </summary>
    public void Trim(int maxLen)
    {
        for (int i = 0; i < _keyCache.Count; i++)
        {
            if (_keyCache[i] is null) continue;
            var currLen = _keyCache[i].shape[2];
            if (currLen > maxLen)
            {
                int startIdx = (int)(currLen - maxLen);
                var newKey = _keyCache[i][.., .., startIdx.., ..].clone();
                var newVal = _valueCache[i][.., .., startIdx.., ..].clone();
                _keyCache[i].Dispose();
                _valueCache[i].Dispose();
                _keyCache[i] = newKey;
                _valueCache[i] = newVal;
            }
        }
        SeenTokens = (int)GetSeqLength(0);
    }

    /// <summary>Disposes all cached tensors.</summary>
    public void Dispose()
    {
        for (int i = 0; i < _keyCache.Count; i++)
        {
            _keyCache[i]?.Dispose();
            _valueCache[i]?.Dispose();
        }
        for (int i = 0; i < _convStates.Count; i++)
        {
            _convStates[i]?.Dispose();
            _recurrentStates[i]?.Dispose();
        }
        _keyCache.Clear();
        _valueCache.Clear();
        _convStates.Clear();
        _recurrentStates.Clear();
    }
}
