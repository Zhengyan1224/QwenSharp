using System.Text;

namespace Zhengyan.QwenSharp.Omni.Audio;

public static class ManagedSpeechSynthesizer
{
    private const int SampleRate = 24_000;
    private const float TwoPi = MathF.PI * 2f;

    private static readonly Dictionary<char, string> Mandarin = new()
    {
        ['\u4f60'] = "ni3",
        ['\u597d'] = "hao3",
        ['\u6211'] = "wo3",
        ['\u662f'] = "shi4",
        ['\u7684'] = "de5",
        ['\u4e86'] = "le5",
        ['\u5417'] = "ma5",
        ['\u5462'] = "ne5",
        ['\u5728'] = "zai4",
        ['\u6709'] = "you3",
        ['\u4e0d'] = "bu4",
        ['\u5f88'] = "hen3",
        ['\u9ad8'] = "gao1",
        ['\u5174'] = "xing4",
        ['\u89c1'] = "jian4",
        ['\u5230'] = "dao4",
        ['\u8c22'] = "xie4",
        ['\u8bf7'] = "qing3",
        ['\u95ee'] = "wen4",
        ['\u53ef'] = "ke3",
        ['\u4ee5'] = "yi3",
        ['\u5e2e'] = "bang1",
        ['\u60a8'] = "nin2",
        ['\u4ec0'] = "shen2",
        ['\u4e48'] = "me5",
        ['\u4eca'] = "jin1",
        ['\u5929'] = "tian1",
        ['\u6c14'] = "qi4",
        ['\u8bf4'] = "shuo1",
        ['\u542c'] = "ting1",
        ['\u770b'] = "kan4",
        ['\u8fd9'] = "zhe4",
        ['\u90a3'] = "na4",
        ['\u4e2a'] = "ge4",
        ['\u5bf9'] = "dui4",
        ['\u6ca1'] = "mei2",
        ['\u9898'] = "ti2",
        ['\u5c31'] = "jiu4",
        ['\u80fd'] = "neng2",
        ['\u548c'] = "he2",
        ['\u6587'] = "wen2",
        ['\u5b57'] = "zi4",
        ['\u4ea4'] = "jiao1",
        ['\u6d41'] = "liu2",
        ['\u6b22'] = "huan1",
        ['\u8fce'] = "ying2",
        ['\u4f7f'] = "shi3",
        ['\u7528'] = "yong4",
    };

    public static string SynthesizeToWavBase64(string text, string? voice)
    {
        var samples = Synthesize(text, voice);
        var bytes = WavCodec.WritePcm16Wav(samples, SampleRate);
        return Convert.ToBase64String(bytes);
    }

    private static float[] Synthesize(string text, string? voice)
    {
        var profile = ResolveProfile(voice);
        var units = Tokenize(text);
        var output = new List<float>(Math.Max(1, text.Length) * SampleRate / 3);
        var carryPhase = 0f;

        foreach (var unit in units)
        {
            if (unit.IsPause)
            {
                AppendSilence(output, unit.Duration);
                continue;
            }

            AppendSyllable(output, unit, profile, ref carryPhase);
        }

        if (output.Count == 0)
        {
            var fallback = new SpeechUnit("e5", "e", 5, 0.34f, false);
            AppendSyllable(output, fallback, profile, ref carryPhase);
        }

        ApplyLimiter(output);
        FadeEdges(output, 0.012f);
        return output.ToArray();
    }

    private static VoiceProfile ResolveProfile(string? voice)
    {
        if (string.Equals(voice, "Ethan", StringComparison.OrdinalIgnoreCase))
        {
            return new VoiceProfile(BasePitch: 128f, Speed: 0.92f, Breath: 0.010f, Brightness: 0.88f);
        }

        if (string.Equals(voice, "Cherry", StringComparison.OrdinalIgnoreCase))
        {
            return new VoiceProfile(BasePitch: 218f, Speed: 1.02f, Breath: 0.008f, Brightness: 1.08f);
        }

        if (string.Equals(voice, "Serena", StringComparison.OrdinalIgnoreCase))
        {
            return new VoiceProfile(BasePitch: 198f, Speed: 0.90f, Breath: 0.008f, Brightness: 0.96f);
        }

        return new VoiceProfile(BasePitch: 205f, Speed: 0.96f, Breath: 0.008f, Brightness: 1.0f);
    }

    private static IReadOnlyList<SpeechUnit> Tokenize(string text)
    {
        var units = new List<SpeechUnit>();
        var word = new StringBuilder();

        foreach (var ch in text)
        {
            if (char.IsLetterOrDigit(ch) && ch < 128)
            {
                word.Append(char.ToLowerInvariant(ch));
                continue;
            }

            FlushWord(word, units);

            if (char.IsWhiteSpace(ch))
            {
                AddPause(units, 0.05f);
            }
            else if (IsStrongPunctuation(ch))
            {
                AddPause(units, 0.28f);
            }
            else if (IsLightPunctuation(ch))
            {
                AddPause(units, 0.14f);
            }
            else if (IsCjk(ch))
            {
                units.Add(ParseMandarin(ch));
                AddPause(units, 0.018f);
            }
        }

        FlushWord(word, units);
        return units;
    }

    private static void FlushWord(StringBuilder word, List<SpeechUnit> units)
    {
        if (word.Length == 0)
        {
            return;
        }

        var text = word.ToString();
        word.Clear();
        var vowel = ExtractEnglishVowel(text);
        var duration = Math.Clamp(0.20f + text.Length * 0.022f, 0.24f, 0.48f);
        units.Add(new SpeechUnit(text, vowel, 5, duration, false));
        AddPause(units, 0.04f);
    }

    private static SpeechUnit ParseMandarin(char ch)
    {
        if (!Mandarin.TryGetValue(ch, out var pinyin))
        {
            pinyin = FallbackPinyin(ch);
        }

        var tone = char.IsDigit(pinyin[^1]) ? pinyin[^1] - '0' : 5;
        var body = char.IsDigit(pinyin[^1]) ? pinyin[..^1] : pinyin;
        return new SpeechUnit(pinyin, ExtractMandarinFinal(body), tone, 0.31f, false);
    }

    private static string FallbackPinyin(char ch)
    {
        var finals = new[] { "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "i", "ian", "iao", "ie", "in", "ing", "o", "ong", "ou", "u", "uan", "uo" };
        var code = (int)ch;
        return finals[code % finals.Length] + ((code % 4) + 1).ToString();
    }

    private static string ExtractMandarinFinal(string pinyin)
    {
        var finals = new[] { "iang", "iong", "uang", "ang", "eng", "ian", "iao", "ing", "ong", "uai", "uan", "ai", "an", "ao", "ei", "en", "er", "ia", "ie", "in", "iu", "ou", "ua", "ue", "ui", "un", "uo", "a", "e", "i", "o", "u", "v" };
        foreach (var final in finals)
        {
            if (pinyin.EndsWith(final, StringComparison.Ordinal))
            {
                return final;
            }
        }

        return "e";
    }

    private static string ExtractEnglishVowel(string word)
    {
        if (word.Contains("oo", StringComparison.Ordinal)) return "u";
        if (word.Contains("ee", StringComparison.Ordinal) || word.Contains("ea", StringComparison.Ordinal)) return "i";
        if (word.Contains("ou", StringComparison.Ordinal) || word.Contains("ow", StringComparison.Ordinal)) return "ao";
        if (word.Contains("ai", StringComparison.Ordinal) || word.Contains("ay", StringComparison.Ordinal)) return "ai";
        if (word.Contains("a", StringComparison.Ordinal)) return "a";
        if (word.Contains("i", StringComparison.Ordinal)) return "i";
        if (word.Contains("o", StringComparison.Ordinal)) return "o";
        if (word.Contains("u", StringComparison.Ordinal)) return "u";
        return "e";
    }

    private static void AppendSyllable(List<float> output, SpeechUnit unit, VoiceProfile profile, ref float phase)
    {
        var duration = unit.Duration / profile.Speed;
        var count = Math.Max(1, (int)Math.Round(duration * SampleRate));
        var formants = GetFormants(unit.Vowel);
        var f1 = new Resonator(formants.F1 * profile.Brightness, formants.B1, SampleRate);
        var f2 = new Resonator(formants.F2 * profile.Brightness, formants.B2, SampleRate);
        var f3 = new Resonator(formants.F3 * profile.Brightness, formants.B3, SampleRate);
        var noiseState = (uint)(unit.Text.GetHashCode(StringComparison.Ordinal) | 1);

        for (var i = 0; i < count; i++)
        {
            var t = i / (float)Math.Max(1, count - 1);
            var pitch = PitchAt(profile.BasePitch, unit.Tone, t);
            phase += TwoPi * pitch / SampleRate;
            if (phase > TwoPi)
            {
                phase -= TwoPi;
            }

            noiseState = noiseState * 1664525u + 1013904223u;
            var noise = (((noiseState >> 8) & 0xffff) / 32768f) - 1f;
            var glottal = MathF.Sin(phase)
                + 0.26f * MathF.Sin(2f * phase)
                + 0.08f * MathF.Sin(3f * phase);

            var onsetNoise = t < 0.10f && HasNoisyOnset(unit.Text)
                ? noise * (0.10f * (1f - t / 0.10f))
                : 0f;

            var source = glottal * 0.66f + noise * profile.Breath + onsetNoise;
            var vowel = f1.Process(source) * 0.46f
                + f2.Process(source) * 0.30f
                + f3.Process(source) * 0.12f;

            var sample = SoftClip(vowel * Envelope(t) * 0.95f);
            output.Add(sample);
        }
    }

    private static float PitchAt(float basePitch, int tone, float t)
        => tone switch
        {
            1 => basePitch * (1.08f - 0.02f * t),
            2 => basePitch * (0.90f + 0.28f * t),
            3 => basePitch * (t < 0.58f ? 0.98f - 0.20f * (t / 0.58f) : 0.78f + 0.16f * ((t - 0.58f) / 0.42f)),
            4 => basePitch * (1.18f - 0.34f * t),
            _ => basePitch * (0.98f + 0.03f * MathF.Sin(MathF.PI * t)),
        };

    private static float Envelope(float t)
    {
        var attack = SmoothStep(Math.Clamp(t / 0.065f, 0f, 1f));
        var release = SmoothStep(Math.Clamp((1f - t) / 0.12f, 0f, 1f));
        return MathF.Min(attack, release);
    }

    private static float SmoothStep(float x) => x * x * (3f - (2f * x));

    private static float SoftClip(float x) => x / (1f + MathF.Abs(x));

    private static bool HasNoisyOnset(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        return text.StartsWith("s", StringComparison.Ordinal)
            || text.StartsWith("sh", StringComparison.Ordinal)
            || text.StartsWith("x", StringComparison.Ordinal)
            || text.StartsWith("h", StringComparison.Ordinal)
            || text.StartsWith("f", StringComparison.Ordinal)
            || text.StartsWith("ch", StringComparison.Ordinal)
            || text.StartsWith("q", StringComparison.Ordinal);
    }

    private static (float F1, float B1, float F2, float B2, float F3, float B3) GetFormants(string vowel)
        => vowel switch
        {
            "a" or "ang" or "an" or "ia" or "uan" => (760f, 120f, 1250f, 170f, 2550f, 250f),
            "ai" or "ei" => (620f, 110f, 1820f, 180f, 2720f, 260f),
            "ao" or "o" or "ong" or "uo" => (500f, 105f, 980f, 150f, 2350f, 240f),
            "e" or "en" or "eng" or "er" => (470f, 105f, 1620f, 170f, 2460f, 250f),
            "i" or "in" or "ing" or "ian" or "ie" => (300f, 90f, 2180f, 190f, 2980f, 260f),
            "u" or "ou" or "ua" or "ui" or "un" => (350f, 90f, 900f, 150f, 2200f, 240f),
            "v" or "ue" => (320f, 95f, 1500f, 170f, 2380f, 250f),
            _ => (500f, 105f, 1450f, 170f, 2450f, 250f),
        };

    private static void AddPause(List<SpeechUnit> units, float duration)
    {
        if (units.Count > 0 && units[^1].IsPause)
        {
            units[^1] = units[^1] with { Duration = Math.Max(units[^1].Duration, duration) };
            return;
        }

        units.Add(new SpeechUnit("", "", 5, duration, true));
    }

    private static void AppendSilence(List<float> output, float duration)
    {
        var count = Math.Max(1, (int)Math.Round(duration * SampleRate));
        for (var i = 0; i < count; i++)
        {
            output.Add(0f);
        }
    }

    private static void ApplyLimiter(List<float> samples)
    {
        var peak = 0f;
        foreach (var sample in samples)
        {
            peak = Math.Max(peak, MathF.Abs(sample));
        }

        if (peak < 1e-6f)
        {
            return;
        }

        var gain = 0.86f / peak;
        for (var i = 0; i < samples.Count; i++)
        {
            samples[i] = Math.Clamp(samples[i] * gain, -0.92f, 0.92f);
        }
    }

    private static void FadeEdges(List<float> samples, float seconds)
    {
        var count = Math.Min(samples.Count / 2, Math.Max(1, (int)(seconds * SampleRate)));
        for (var i = 0; i < count; i++)
        {
            var gain = i / (float)count;
            samples[i] *= gain;
            samples[samples.Count - 1 - i] *= gain;
        }
    }

    private static bool IsCjk(char ch) => ch >= '\u3400' && ch <= '\u9fff';

    private static bool IsStrongPunctuation(char ch)
        => ch is '.' or '!' or '?' or '\u3002' or '\uff01' or '\uff1f' or '\n';

    private static bool IsLightPunctuation(char ch)
        => ch is ',' or ';' or ':' or '\uff0c' or '\uff1b' or '\uff1a' or '\u3001';

    private readonly record struct SpeechUnit(string Text, string Vowel, int Tone, float Duration, bool IsPause);

    private readonly record struct VoiceProfile(float BasePitch, float Speed, float Breath, float Brightness);

    private sealed class Resonator
    {
        private readonly float _a0;
        private readonly float _b1;
        private readonly float _b2;
        private float _y1;
        private float _y2;

        public Resonator(float frequency, float bandwidth, int sampleRate)
        {
            var radius = MathF.Exp(-MathF.PI * bandwidth / sampleRate);
            var theta = TwoPi * frequency / sampleRate;
            _a0 = 1f - radius;
            _b1 = 2f * radius * MathF.Cos(theta);
            _b2 = -(radius * radius);
        }

        public float Process(float input)
        {
            var y = (_a0 * input) + (_b1 * _y1) + (_b2 * _y2);
            _y2 = _y1;
            _y1 = y;
            return y;
        }
    }
}
