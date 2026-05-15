using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Common;

/// <summary>
/// Qwen2 RMSNorm implementation.
/// Mathematically equivalent to LlamaRMSNorm.
/// 
/// NOTE: Weight is accessed via get_parameter("weight") at forward time rather than
/// using a cached C# field, because TorchSharp's module.cuda() replaces parameter
/// objects in the internal registry without updating the cached C# field reference.
/// </summary>
public class RMSNorm : Module<Tensor, Tensor>
{
    private readonly double _eps;

    public RMSNorm(long hiddenSize, double eps = 1e-6, string name = "RMSNorm") : base(name)
    {
        _eps = eps;
        
        var w = torch.ones(hiddenSize);
        register_parameter("weight", new Parameter(w));
    }

    public override Tensor forward(Tensor x)
    {
        var inputDtype = x.dtype;
        
        // Cast to FP32 for numerical stability (same as HF implementation)
        var xFp32 = x.to_type(ScalarType.Float32);
        
        // variance = x.pow(2).mean(-1, keepdim=True)
        var variance = xFp32.pow(2).mean(new long[] { -1 }, keepdim: true);
        
        // x = x * torch.rsqrt(variance + eps)
        var rsqrt = (variance + _eps).rsqrt();
        var normedFp32 = xFp32 * rsqrt;
        
        // Cast back to original dtype
        var normed = normedFp32.to_type(inputDtype);
        
        // Access the weight parameter live from the module, so we always get the
        // correct CUDA parameter after model.cuda() has been called.
        var weight = get_parameter("weight");
        return weight * normed;
    }
}

