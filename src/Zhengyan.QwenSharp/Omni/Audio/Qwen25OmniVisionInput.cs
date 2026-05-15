using TorchSharp;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.Omni.Audio;

public sealed record Qwen25OmniVisionInput(
    string Path,
    string Kind,
    Tensor PixelValues,
    Tensor ImageGridThw) : IDisposable
{
    public void Dispose()
    {
        PixelValues.Dispose();
        ImageGridThw.Dispose();
    }
}
