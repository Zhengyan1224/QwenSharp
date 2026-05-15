using System.Diagnostics;
using TorchSharp;
using Zhengyan.QwenSharp.Core;
using Zhengyan.QwenSharp.Generation;
using Zhengyan.QwenSharp.Models;
using Zhengyan.QwenSharp.Tokenizers;
using Zhengyan.QwenSharp.Vision;
using static TorchSharp.torch;

namespace Zhengyan.QwenSharp.Cli;

class Program
{
    static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            PrintHelp();
            return 1;
        }

        string command = args[0].ToLowerInvariant();
        string? modelPath = GetArgValue(args, "--model-path");

        if (string.IsNullOrEmpty(modelPath))
        {
            Console.WriteLine("Error: --model-path is required.");
            PrintHelp();
            return 1;
        }

        try
        {
            switch (command)
            {
                case "chat":
                    RunChatCommand(modelPath);
                    break;
                case "generate":
                    {
                        var prompt = GetPositionalArg(args, 1);
                        if (string.IsNullOrEmpty(prompt))
                        {
                            Console.WriteLine("Error: generate requires a prompt. e.g. generate \"Hello world\" --model-path ...");
                            return 1;
                        }

                        RunGenerateCommand(modelPath, prompt);
                        break;
                    }
                case "benchmark":
                    {
                        var prompt = GetPositionalArg(args, 1);
                        if (string.IsNullOrEmpty(prompt))
                        {
                            Console.WriteLine("Error: benchmark requires a prompt. e.g. benchmark \"Hello world\" --model-path ...");
                            return 1;
                        }

                        RunBenchmarkCommand(modelPath, prompt);
                        break;
                    }
                case "image-prompt":
                    {
                        var imagePath = GetArgValue(args, "--image-path");
                        if (string.IsNullOrEmpty(imagePath))
                        {
                            Console.WriteLine("Error: image-prompt requires --image-path <path>.");
                            return 1;
                        }

                        var instruction = GetArgValue(args, "--instruction")
                                          ?? "Generate a concise, high-quality English image generation prompt for this image. Focus on subject, style, lighting, composition, colors, and important details. Output prompt only.";
                        RunImagePromptCommand(modelPath, imagePath, instruction);
                        break;
                    }
                case "judge-image":
                    {
                        var imagePath = GetArgValue(args, "--image-path");
                        var referencePrompt = GetArgValue(args, "--reference-prompt");
                        if (string.IsNullOrEmpty(imagePath) || string.IsNullOrEmpty(referencePrompt))
                        {
                            Console.WriteLine("Error: judge-image requires --image-path <path> and --reference-prompt <prompt>.");
                            return 1;
                        }

                        RunJudgeImageCommand(modelPath, imagePath, referencePrompt);
                        break;
                    }
                default:
                    Console.WriteLine($"Unknown command: {command}");
                    PrintHelp();
                    return 1;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error during execution: {ex.Message}\n{ex.StackTrace}");
            return 1;
        }

        return 0;
    }

    private static string? GetArgValue(string[] args, string flag)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == flag)
            {
                return args[i + 1];
            }
        }

        return null;
    }

    private static string? GetPositionalArg(string[] args, int index)
    {
        var positionalArgs = args.Skip(1).Where(a => !a.StartsWith("--")).ToList();
        if (index - 1 >= 0 && index - 1 < positionalArgs.Count)
        {
            int pos = 0;
            for (int i = 1; i < args.Length; i++)
            {
                if (args[i].StartsWith("--"))
                {
                    i++;
                    continue;
                }

                pos++;
                if (pos == index)
                {
                    return args[i];
                }
            }
        }

        return null;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("Zhengyan.QwenSharp CLI");
        Console.WriteLine("Usage: dotnet run --project samples/Zhengyan.QwenSharp.Cli -- <command> [prompt] --model-path <path>");
        Console.WriteLine();
        Console.WriteLine("Commands:");
        Console.WriteLine("  chat          Start an interactive chat session");
        Console.WriteLine("  generate      Generate text based on a prompt");
        Console.WriteLine("  benchmark     Run generation and measure tokens/second");
        Console.WriteLine("  image-prompt  Generate an image prompt from a local image using a VL model");
        Console.WriteLine("  judge-image   Judge whether a local image matches a reference prompt using a VL model");
        Console.WriteLine("  voice-demo    Run the browser-based Qwen2.5-Omni realtime demo from samples/Zhengyan.QwenSharp.OmniWebDemo");
    }

    private static void EnsureLibTorch()
    {
        torch.InitializeDeviceType(TorchHelper.IsCudaAvailable() ? DeviceType.CUDA : DeviceType.CPU);
    }

    private static Device GetDevice()
        => TorchHelper.GetDefaultDevice();

    private static void RunGenerateCommand(string modelPath, string prompt)
    {
        EnsureLibTorch();
        var device = GetDevice();
        Console.WriteLine($"Loading model from {modelPath}...");

        var tokenizer = Qwen2Tokenizer.FromDirectory(modelPath);
        var model = ModelLoader.LoadModel(modelPath, out _);
        ((TorchSharp.torch.nn.Module)model).to(device);

        var inputIdsList = tokenizer.Encode(prompt);
        Console.WriteLine($"Token IDs: {string.Join(", ", inputIdsList)}");
        using var inputIds = torch.tensor(inputIdsList, dtype: ScalarType.Int64).unsqueeze(0).to(device);

        var config = new GenerationConfig { MaxNewTokens = 512, Temperature = 0.7f, TopP = 0.8f, DoSample = true };

        Console.WriteLine($"Prompt: {prompt}\n---");

        foreach (var tokenId in TextGenerator.Generate(model, inputIds, config, tokenizer.EosTokenId ?? -1, tokenizer.ImEndId))
        {
            var text = tokenizer.Decode(new[] { tokenId }, skipSpecialTokens: true);
            Console.Write(text);
        }

        Console.WriteLine("\n---");
    }

    private static void RunChatCommand(string modelPath)
    {
        EnsureLibTorch();
        var device = GetDevice();
        Console.WriteLine($"Loading model from {modelPath}...");

        var tokenizer = Qwen2Tokenizer.FromDirectory(modelPath);
        var model = ModelLoader.LoadModel(modelPath, out _);
        ((TorchSharp.torch.nn.Module)model).to(device);

        GC.Collect();
        GC.WaitForPendingFinalizers();

        var messages = new List<ChatMessage>();
        var config = new GenerationConfig { MaxNewTokens = 1024, Temperature = 0.7f, TopP = 0.8f, DoSample = true };

        Console.WriteLine("Interactive Chat started. Type 'exit' or 'quit' to end.");
        while (true)
        {
            Console.Write("User: ");
            var input = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(input))
            {
                continue;
            }

            if (input.Equals("exit", StringComparison.OrdinalIgnoreCase) || input.Equals("quit", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            messages.Add(new ChatMessage("user", input));

            var promptIds = tokenizer.EncodeChatTemplate(messages);
            using var inputIds = torch.tensor(promptIds, dtype: ScalarType.Int64).unsqueeze(0).to(device);

            Console.Write("Assistant: ");
            var responseIds = new List<int>();

            foreach (var tokenId in TextGenerator.Generate(model, inputIds, config, tokenizer.EosTokenId ?? -1, tokenizer.ImEndId))
            {
                var text = tokenizer.Decode(new[] { tokenId }, skipSpecialTokens: true);
                Console.Write(text);
                responseIds.Add(tokenId);
            }

            Console.WriteLine();

            var fullResponse = tokenizer.Decode(responseIds, skipSpecialTokens: true);
            messages.Add(new ChatMessage("assistant", fullResponse));
        }
    }

    private static void RunBenchmarkCommand(string modelPath, string prompt)
    {
        EnsureLibTorch();
        var device = GetDevice();
        Console.WriteLine($"Loading model from {modelPath}...");

        var tokenizer = Qwen2Tokenizer.FromDirectory(modelPath);
        var model = ModelLoader.LoadModel(modelPath, out _);
        ((TorchSharp.torch.nn.Module)model).to(device);

        GC.Collect();
        GC.WaitForPendingFinalizers();

        var inputIdsList = tokenizer.Encode(prompt);
        Console.WriteLine($"Token IDs: {string.Join(", ", inputIdsList)}");
        using var inputIds = torch.tensor(inputIdsList, dtype: ScalarType.Int64).unsqueeze(0).to(device);

        var config = new GenerationConfig { MaxNewTokens = 100, DoSample = false };

        Console.WriteLine("Warming up...");
        foreach (var _ in TextGenerator.Generate(model, inputIds, new GenerationConfig { MaxNewTokens = 2, DoSample = false }, tokenizer.EosTokenId ?? -1, tokenizer.ImEndId))
        {
        }

        Console.WriteLine("Benchmarking generation speed...");
        var sw = Stopwatch.StartNew();

        int tokenCount = 0;
        foreach (var _ in TextGenerator.Generate(model, inputIds, config, tokenizer.EosTokenId ?? -1, tokenizer.ImEndId))
        {
            tokenCount++;
        }

        sw.Stop();

        Console.WriteLine($"Generated {tokenCount} tokens in {sw.Elapsed.TotalSeconds:F2} seconds.");
        if (tokenCount > 0)
        {
            Console.WriteLine($"Speed: {tokenCount / sw.Elapsed.TotalSeconds:F2} tokens/sec.");
        }
    }

    private static void RunImagePromptCommand(string modelPath, string imagePath, string instruction)
    {
        EnsureLibTorch();
        var device = GetDevice();
        Console.WriteLine($"Loading model from {modelPath}...");

        var tokenizer = Qwen2Tokenizer.FromDirectory(modelPath);
        var model = ModelLoader.LoadVisionLanguageModel(modelPath, out _);
        ((TorchSharp.torch.nn.Module)model).to(device);

        using var visionInput = QwenVisionProcessor.PrepareImagePrompt(modelPath, tokenizer, imagePath, instruction);
        using var inputIds = torch.tensor(visionInput.InputIds, dtype: ScalarType.Int64).unsqueeze(0).to(device);
        using var pixelValues = visionInput.PixelValues.to(device);
        using var imageGridThw = visionInput.ImageGridThw.to(device);

        GC.Collect();
        GC.WaitForPendingFinalizers();

        var config = new GenerationConfig
        {
            MaxNewTokens = 256,
            Temperature = 0.0f,
            TopP = 1.0f,
            TopK = 0,
            DoSample = false
        };

        Console.Write("Assistant: ");
        foreach (var tokenId in MultimodalTextGenerator.Generate(model, inputIds, pixelValues, imageGridThw, config, tokenizer.EosTokenId ?? -1, tokenizer.ImEndId))
        {
            var text = tokenizer.Decode(new[] { tokenId }, skipSpecialTokens: true);
            Console.Write(text);
        }

        Console.WriteLine();
    }

    private static void RunJudgeImageCommand(string modelPath, string imagePath, string referencePrompt)
    {
        EnsureLibTorch();
        var device = GetDevice();
        Console.WriteLine($"Loading model from {modelPath}...");

        var tokenizer = Qwen2Tokenizer.FromDirectory(modelPath);
        var model = ModelLoader.LoadVisionLanguageModel(modelPath, out _);
        ((TorchSharp.torch.nn.Module)model).to(device);

        var instruction = $$"""
请判断这张图片是否符合下面的参考描述。
参考描述：
{{referencePrompt}}

请仅返回 JSON，格式如下：
{"verdict":"match|partial|mismatch","score":0,"reason":"简要说明"}
""";

        using var visionInput = QwenVisionProcessor.PrepareImagePrompt(modelPath, tokenizer, imagePath, instruction);
        using var inputIds = torch.tensor(visionInput.InputIds, dtype: ScalarType.Int64).unsqueeze(0).to(device);
        using var pixelValues = visionInput.PixelValues.to(device);
        using var imageGridThw = visionInput.ImageGridThw.to(device);

        GC.Collect();
        GC.WaitForPendingFinalizers();

        var config = new GenerationConfig
        {
            MaxNewTokens = 128,
            Temperature = 0.0f,
            TopP = 1.0f,
            TopK = 0,
            DoSample = false
        };

        Console.Write("Assistant: ");
        foreach (var tokenId in MultimodalTextGenerator.Generate(model, inputIds, pixelValues, imageGridThw, config, tokenizer.EosTokenId ?? -1, tokenizer.ImEndId))
        {
            var text = tokenizer.Decode(new[] { tokenId }, skipSpecialTokens: true);
            Console.Write(text);
        }

        Console.WriteLine();
    }
}
