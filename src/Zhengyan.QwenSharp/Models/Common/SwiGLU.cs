using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Common;

/// <summary>
/// SwiGLU MLP implementation used in Qwen models.
/// Computes: down_proj(silu(gate_proj(x)) * up_proj(x))
/// </summary>
public class SwiGLU : Module<Tensor, Tensor>
{
    private readonly Linear _gateProj;
    private readonly Linear _upProj;
    private readonly Linear _downProj;
    private readonly SiLU _actFn;

    public SwiGLU(long hiddenSize, long intermediateSize, bool bias = false, string name = "SwiGLU") : base(name)
    {
        _gateProj = Linear(hiddenSize, intermediateSize, hasBias: bias);
        _upProj = Linear(hiddenSize, intermediateSize, hasBias: bias);
        _downProj = Linear(intermediateSize, hiddenSize, hasBias: bias);
        _actFn = SiLU();
        
        register_module("gate_proj", _gateProj);
        register_module("up_proj", _upProj);
        register_module("down_proj", _downProj);
        register_module("act_fn", _actFn);
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        
        var gate = _gateProj.forward(x);
        var act = _actFn.forward(gate);
        
        var up = _upProj.forward(x);
        var intermediate = act * up;
        
        var output = _downProj.forward(intermediate);
        
        return scope.MoveToOuter(output);
    }
}
