using TorchSharp;
using TorchSharp.Modules;
using Zhengyan.QwenSharp.Core;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Zhengyan.QwenSharp.Models.Qwen25Omni;

/// <summary>
/// Qwen2.5-Omni Talker Model. 
/// Generates semantic and acoustic tokens from text/audio/video conditions,
/// and converts them directly into audio waveforms using the internal Token2Wav Vocoder.
/// </summary>
public class Qwen25OmniTalkerForConditionalGeneration : nn.Module
{
    private readonly Qwen25OmniThinkerForConditionalGeneration _thinker;
    private readonly Qwen25OmniToken2WavModel _token2Wav;

    public Qwen25OmniTalkerForConditionalGeneration(Qwen25OmniConfig config, string name = "model") : base(name)
    {
        _thinker = new Qwen25OmniThinkerForConditionalGeneration(config.ThinkerConfig, name: "thinker");
        
        var token2WavConfig = new Qwen25OmniToken2WavConfig();
        _token2Wav = new Qwen25OmniToken2WavModel(token2WavConfig, name: "token2wav");

        register_module("thinker", _thinker);
        register_module("token2wav", _token2Wav);
    }

    /// <summary>
    /// Forward pass generating audio waveforms given multimodal input contexts.
    /// </summary>
    public Tensor forward(Tensor inputIds, Tensor? inputFeatures = null, Tensor? videoFeatures = null, Tensor? imageGridThw = null, Tensor? positionIds = null, Zhengyan.QwenSharp.Models.Common.KVCache? kvCache = null)
    {
        using var scope = torch.NewDisposeScope();
        
        // 1. Compute contextual representation using the Thinker
        Tensor logits;
        if (inputFeatures is not null) 
        {
            logits = _thinker.forward(inputIds, inputFeatures, positionIds, kvCache);
        }
        else 
        {
            logits = _thinker.forward(inputIds, positionIds, kvCache);
        }
        
        // 2. Synthesize audio waveform using the vocoder diffusion/GAN model
        var audioWaveform = _token2Wav.forward(logits);
        
        return scope.MoveToOuter(audioWaveform);
    }

    public static Qwen25OmniTalkerForConditionalGeneration FromPretrained(string modelDirectory)
    {
        var config = ModelConfig.FromDirectory<Qwen25OmniConfig>(modelDirectory);
        var model = new Qwen25OmniTalkerForConditionalGeneration(config);
        
        var stateDict = SafeTensorsLoader.LoadFromDirectory(modelDirectory);
        model.load_state_dict(stateDict, strict: false);
        
        foreach (var tensor in stateDict.Values) tensor.Dispose();
        return model;
    }
}
