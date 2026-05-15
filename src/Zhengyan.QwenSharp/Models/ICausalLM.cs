using TorchSharp;
using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.Models;

/// <summary>
/// Interface for Causal Language Models supporting autoregressive generation.
/// All implementations must be TorchSharp nn.Module instances to support device management (.cuda(), .cpu(), etc.).
/// </summary>
public interface ICausalLM
{
    Tensor forward(Tensor inputIds, Tensor? positionIds = null, KVCache? kvCache = null);
}
