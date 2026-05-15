using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.Omni.Audio;

public static class Qwen25OmniAudioProcessor
{
    private const int TargetSampleRate = 16_000;
    private const int Nfft = 400;
    private const int WinLength = 400;
    private const int HopLength = 160;
    private const int MelBins = 128;
    private const int OmniAudioWindow = 100;
    private const int FftBins = (Nfft / 2) + 1;

    private static readonly float[] Window = BuildHannWindow(WinLength);
    private static readonly float[,] MelFilterBank = BuildMelFilterBank();
    private static readonly float[,] DftCos = BuildDftTable(cosine: true);
    private static readonly float[,] DftSin = BuildDftTable(cosine: false);

    public static Tensor LoadMelSpectrogram(string wavPath)
    {
        var samples = WavCodec.ReadMonoSamples(wavPath, out var sampleRate);
        if (sampleRate != TargetSampleRate)
        {
            samples = ResampleLinear(samples, sampleRate, TargetSampleRate);
        }

        return MelSpectrogramFromSamples(samples);
    }

    public static Tensor MelSpectrogramFromSamples(ReadOnlySpan<float> samples)
    {
        var frameCount = CalculateFrameCount(samples.Length);
        var output = new float[MelBins * frameCount];
        var powerSpectrum = new float[FftBins];
        var maxLog = float.NegativeInfinity;

        for (var frame = 0; frame < frameCount; frame++)
        {
            var offset = (frame * HopLength) - (Nfft / 2);

            for (var bin = 0; bin < powerSpectrum.Length; bin++)
            {
                var real = 0f;
                var imag = 0f;
                for (var i = 0; i < Nfft; i++)
                {
                    var sample = GetReflectedSample(samples, offset + i) * Window[i];
                    real += sample * DftCos[bin, i];
                    imag -= sample * DftSin[bin, i];
                }

                powerSpectrum[bin] = (real * real) + (imag * imag);
            }

            for (var mel = 0; mel < MelBins; mel++)
            {
                var energy = 0f;
                for (var bin = 0; bin < powerSpectrum.Length; bin++)
                {
                    energy += powerSpectrum[bin] * MelFilterBank[mel, bin];
                }

                var logEnergy = MathF.Log10(MathF.Max(energy, 1e-10f));
                output[mel * frameCount + frame] = logEnergy;
                maxLog = Math.Max(maxLog, logEnergy);
            }
        }

        var floor = maxLog - 8f;
        for (var i = 0; i < output.Length; i++)
        {
            output[i] = (Math.Max(output[i], floor) + 4f) / 4f;
        }

        using var scope = NewDisposeScope();
        var baseTensor = tensor(output, dtype: ScalarType.Float32);
        return scope.MoveToOuter(baseTensor.view(1, MelBins, frameCount));
    }

    public static int EstimateAudioTokenCount(Tensor melSpectrogram)
    {
        var frameCount = (int)melSpectrogram.shape[2];
        return EstimateAudioTokenCount(frameCount);
    }

    public static int EstimateAudioTokenCount(int frameCount, int nWindow = OmniAudioWindow)
    {
        frameCount = Math.Max(1, frameCount);
        var chunkFrameLen = Math.Max(1, nWindow * 2);
        var audioAfterCnnLen = 0;

        for (var start = 0; start < frameCount; start += chunkFrameLen)
        {
            var chunkLen = Math.Min(chunkFrameLen, frameCount - start);
            audioAfterCnnLen += GetConvStride2OutputLength(chunkLen);
        }

        if (audioAfterCnnLen < 2)
        {
            return Math.Max(1, audioAfterCnnLen);
        }

        return (audioAfterCnnLen - 2) / 2 + 1;
    }

    private static int CalculateFrameCount(int sampleCount)
        => Math.Max(1, sampleCount / HopLength);

    private static int GetConvStride2OutputLength(int inputLength)
        => Math.Max(1, (inputLength - 1) / 2 + 1);

    private static float[] BuildHannWindow(int length)
    {
        var window = new float[length];
        for (var i = 0; i < length; i++)
        {
            window[i] = 0.5f - 0.5f * MathF.Cos((2f * MathF.PI * i) / length);
        }

        return window;
    }

    private static float[,] BuildMelFilterBank()
    {
        var filters = new float[MelBins, FftBins];
        var melMin = HzToMel(0f);
        var melMax = HzToMel(TargetSampleRate / 2f);
        var melPoints = new float[MelBins + 2];

        for (var i = 0; i < melPoints.Length; i++)
        {
            melPoints[i] = melMin + (melMax - melMin) * i / (MelBins + 1);
        }

        var hzPoints = new float[melPoints.Length];
        for (var i = 0; i < melPoints.Length; i++)
        {
            hzPoints[i] = MelToHz(melPoints[i]);
        }

        var fftFrequencies = new float[FftBins];
        for (var i = 0; i < fftFrequencies.Length; i++)
        {
            fftFrequencies[i] = i * (TargetSampleRate / 2f) / (FftBins - 1);
        }

        var fdiff = new float[hzPoints.Length - 1];
        for (var i = 0; i < fdiff.Length; i++)
        {
            fdiff[i] = hzPoints[i + 1] - hzPoints[i];
        }

        for (var i = 0; i < hzPoints.Length; i++)
        {
            if (i < MelBins)
            {
                var enorm = 2f / Math.Max(1e-10f, hzPoints[i + 2] - hzPoints[i]);
                for (var bin = 0; bin < FftBins; bin++)
                {
                    var lower = -(hzPoints[i] - fftFrequencies[bin]) / Math.Max(1e-10f, fdiff[i]);
                    var upper = (hzPoints[i + 2] - fftFrequencies[bin]) / Math.Max(1e-10f, fdiff[i + 1]);
                    filters[i, bin] = Math.Max(0f, Math.Min(lower, upper)) * enorm;
                }
            }
        }

        return filters;
    }

    private static float HzToMel(float hz)
    {
        const float fSp = 200f / 3f;
        const float minLogHz = 1000f;
        const float minLogMel = minLogHz / fSp;
        var logStep = MathF.Log(6.4f) / 27f;
        return hz < minLogHz
            ? hz / fSp
            : minLogMel + MathF.Log(hz / minLogHz) / logStep;
    }

    private static float MelToHz(float mel)
    {
        const float fSp = 200f / 3f;
        const float minLogHz = 1000f;
        const float minLogMel = minLogHz / fSp;
        var logStep = MathF.Log(6.4f) / 27f;
        return mel < minLogMel
            ? mel * fSp
            : minLogHz * MathF.Exp(logStep * (mel - minLogMel));
    }

    private static float[] ResampleLinear(float[] samples, int sourceRate, int targetRate)
    {
        if (samples.Length == 0 || sourceRate == targetRate)
        {
            return samples;
        }

        var ratio = sourceRate / (double)targetRate;
        var targetLength = Math.Max(1, (int)Math.Round(samples.Length / ratio));
        var output = new float[targetLength];

        for (var i = 0; i < targetLength; i++)
        {
            var sourceIndex = i * ratio;
            var index0 = (int)Math.Floor(sourceIndex);
            var index1 = Math.Min(index0 + 1, samples.Length - 1);
            var t = (float)(sourceIndex - index0);
            output[i] = samples[index0] * (1f - t) + samples[index1] * t;
        }

        return output;
    }

    private static float[,] BuildDftTable(bool cosine)
    {
        var table = new float[FftBins, Nfft];

        for (var bin = 0; bin < FftBins; bin++)
        {
            for (var i = 0; i < Nfft; i++)
            {
                var angle = 2f * MathF.PI * bin * i / Nfft;
                table[bin, i] = cosine ? MathF.Cos(angle) : MathF.Sin(angle);
            }
        }

        return table;
    }

    private static float GetReflectedSample(ReadOnlySpan<float> samples, int index)
    {
        if (samples.Length == 0)
        {
            return 0f;
        }

        if (samples.Length == 1)
        {
            return samples[0];
        }

        while (index < 0 || index >= samples.Length)
        {
            if (index < 0)
            {
                index = -index;
            }
            else
            {
                index = (2 * samples.Length) - 2 - index;
            }
        }

        return samples[index];
    }
}

public static class WavCodec
{
    public static float[] DecodePcm16Samples(ReadOnlySpan<byte> data)
    {
        if (data.Length == 0)
        {
            return [];
        }

        var sampleCount = data.Length / sizeof(short);
        var samples = new float[sampleCount];
        for (var i = 0; i < sampleCount; i++)
        {
            var value = BitConverter.ToInt16(data[(i * 2)..((i * 2) + 2)]);
            samples[i] = value / 32768f;
        }

        return samples;
    }

    public static float[] ReadMonoSamples(string wavPath, out int sampleRate)
    {
        using var stream = File.OpenRead(wavPath);
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: false);

        if (new string(reader.ReadChars(4)) != "RIFF")
        {
            throw new InvalidDataException("Invalid WAV file: missing RIFF header.");
        }

        _ = reader.ReadInt32();
        if (new string(reader.ReadChars(4)) != "WAVE")
        {
            throw new InvalidDataException("Invalid WAV file: missing WAVE header.");
        }

        ushort audioFormat = 0;
        ushort numChannels = 0;
        ushort bitsPerSample = 0;
        sampleRate = 0;
        byte[]? data = null;

        while (stream.Position < stream.Length)
        {
            var chunkId = new string(reader.ReadChars(4));
            var chunkSize = reader.ReadInt32();
            var chunkStart = stream.Position;

            if (chunkId == "fmt ")
            {
                audioFormat = reader.ReadUInt16();
                numChannels = reader.ReadUInt16();
                sampleRate = reader.ReadInt32();
                _ = reader.ReadInt32();
                _ = reader.ReadUInt16();
                bitsPerSample = reader.ReadUInt16();
            }
            else if (chunkId == "data")
            {
                data = reader.ReadBytes(chunkSize);
            }
            else
            {
                stream.Position += chunkSize;
            }

            if ((chunkSize & 1) == 1)
            {
                stream.Position++;
            }

            stream.Position = Math.Max(stream.Position, chunkStart + chunkSize);
        }

        if (data is null || sampleRate <= 0 || numChannels == 0)
        {
            throw new InvalidDataException("Invalid WAV file: missing format or data chunk.");
        }

        var samples = new List<float>();
        var bytesPerSample = bitsPerSample / 8;
        var frameSize = bytesPerSample * numChannels;

        for (var offset = 0; offset + frameSize <= data.Length; offset += frameSize)
        {
            float sample;
            switch (audioFormat)
            {
                case 1 when bitsPerSample == 16:
                    sample = BitConverter.ToInt16(data, offset) / 32768f;
                    break;
                case 3 when bitsPerSample == 32:
                    sample = BitConverter.ToSingle(data, offset);
                    break;
                default:
                    throw new NotSupportedException($"Unsupported WAV format: format={audioFormat}, bits={bitsPerSample}");
            }

            if (numChannels > 1)
            {
                var sum = sample;
                for (var channel = 1; channel < numChannels; channel++)
                {
                    sum += audioFormat switch
                    {
                        1 when bitsPerSample == 16 => BitConverter.ToInt16(data, offset + channel * bytesPerSample) / 32768f,
                        3 when bitsPerSample == 32 => BitConverter.ToSingle(data, offset + channel * bytesPerSample),
                        _ => 0f,
                    };
                }

                sample = sum / numChannels;
            }

            samples.Add(sample);
        }

        return samples.ToArray();
    }

    public static byte[] WritePcm16Wav(IReadOnlyList<float> samples, int sampleRate)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        var pcmData = new byte[samples.Count * 2];
        for (var i = 0; i < samples.Count; i++)
        {
            var value = Math.Clamp(samples[i], -1f, 1f);
            var int16 = (short)Math.Round(value * short.MaxValue);
            BitConverter.GetBytes(int16).CopyTo(pcmData, i * 2);
        }

        WriteFourCc(writer, "RIFF");
        writer.Write(36 + pcmData.Length);
        WriteFourCc(writer, "WAVE");
        WriteFourCc(writer, "fmt ");
        writer.Write(16);
        writer.Write((ushort)1);
        writer.Write((ushort)1);
        writer.Write(sampleRate);
        writer.Write(sampleRate * 2);
        writer.Write((ushort)2);
        writer.Write((ushort)16);
        WriteFourCc(writer, "data");
        writer.Write(pcmData.Length);
        writer.Write(pcmData);

        return ms.ToArray();
    }

    private static void WriteFourCc(BinaryWriter writer, string value)
    {
        var bytes = Encoding.ASCII.GetBytes(value);
        if (bytes.Length != 4)
        {
            throw new ArgumentException("FourCC must be exactly 4 ASCII characters.", nameof(value));
        }

        writer.Write(bytes);
    }
}
