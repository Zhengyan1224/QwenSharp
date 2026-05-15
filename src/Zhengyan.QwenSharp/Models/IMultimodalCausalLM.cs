using Zhengyan.QwenSharp.Models.Common;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.Models;

public interface IMultimodalCausalLM : ICausalLM
{
    Tensor forward(
        Tensor inputIds,
        Tensor? positionIds = null,
        KVCache? kvCache = null,
        Tensor? pixelValues = null,
        Tensor? imageGridThw = null);
}
