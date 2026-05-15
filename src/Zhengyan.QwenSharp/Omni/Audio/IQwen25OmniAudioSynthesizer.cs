namespace Zhengyan.QwenSharp.Omni.Audio;

public interface IQwen25OmniAudioSynthesizer : IDisposable
{
    string Name { get; }

    string SynthesizeToWavBase64(string text, string? voice = null);
}
