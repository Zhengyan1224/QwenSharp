# Zhengyan.QwenSharp

[中文](README-zh.md) | English

Zhengyan.QwenSharp is a pure C# inference stack for Qwen models. It uses .NET 10 and TorchSharp to load local Qwen checkpoints, run generation, and expose OpenAI-compatible HTTP and Realtime APIs without introducing a Python service layer.

The project is split into a core inference library, an OpenAI protocol adapter, and runnable samples for CLI, browser-based Omni interaction, and a configurable OpenAI-compatible server.

## Features

- Load local Qwen checkpoints from `config.json`, tokenizer files, and SafeTensors or binary weights.
- Run text generation for Qwen causal language models.
- Run Qwen2.5-Omni text, audio, image, and video understanding through the C# Omni adapter.
- Capture and process browser microphone audio for realtime voice interaction.
- Return synthesized speech through a configurable C# TTS backend.
- Expose OpenAI-style `/v1/chat/completions`, `/v1/responses`, and `/v1/realtime` endpoints.
- Parse OpenAI-style multimodal message parts and function/tool call requests.
- Run on Windows, Linux, and macOS, subject to TorchSharp and LibTorch native runtime support.

## Projects

```text
src/
  Zhengyan.QwenSharp/
    Core model loading, tokenizers, generation helpers, Qwen model implementations,
    vision utilities, and Qwen2.5-Omni audio/vision processing.

  Zhengyan.QwenSharp.OpenAI/
    OpenAI-compatible protocol layer. This project maps HTTP/WebSocket requests to
    QwenSharp inference services and owns the OpenAI request/response contracts.

samples/
  Zhengyan.QwenSharp.Cli/
    Command-line sample for local generation and small utility workflows.

  Zhengyan.QwenSharp.OmniWebDemo/
    Browser demo for Qwen2.5-Omni realtime voice, image, and video interaction.

  Zhengyan.QwenSharp.Server/
    Configurable OpenAI-compatible server. It can host a normal Qwen text model or
    a Qwen2.5-Omni model. Realtime is enabled automatically for Omni models.
```

## Supported Model Families

| Family | Capability | Status |
| --- | --- | --- |
| Qwen2 | Text generation | Supported |
| Qwen2-MoE | Text generation | Supported |
| Qwen3 | Text generation | Supported |
| Qwen3-MoE | Text generation | Supported |
| Qwen3.5 | Text generation | Supported |
| Qwen3.5-MoE | Text generation | Supported |
| Qwen2-VL | Vision-language model classes | Core implementation available |
| Qwen2.5-VL | Vision-language model classes | Core implementation available |
| Qwen3-VL | Vision-language model classes | Core implementation available |
| Qwen2-Audio | Audio-language model classes | Core implementation available |
| Qwen2.5-Omni | Text, audio, image, video, Realtime | Supported through the Omni service path |
| Classification and QA heads | Sequence classification, token classification, question answering | Supported |
| ColQwen2 | Retrieval model | Supported |

The generic `Zhengyan.QwenSharp.Server` text service is text-only for non-Omni models. Multimodal OpenAI request handling is currently wired through the Qwen2.5-Omni service path.

## Requirements

- .NET 10 SDK
- TorchSharp-compatible native runtime
- Local Qwen model directory
- CUDA-capable GPU recommended for larger models
- Optional Sherpa-ONNX TTS model files if you want speech output

This project does not require a Python application process. On Linux, TorchSharp still needs compatible LibTorch or PyTorch native libraries available to the dynamic loader.

## Build

```bash
dotnet build Zhengyan.QwenSharp.slnx
```

## Model Directory

Point `ModelPath` to the directory that contains the model configuration, tokenizer files, and weights.

```text
Models/
  Qwen2.5-Omni-7B/
    config.json
    tokenizer.json
    tokenizer_config.json
    *.safetensors
    *.bin
```

Model files can come from the official Qwen repositories:

- https://huggingface.co/Qwen
- https://github.com/QwenLM

## Run the OpenAI-Compatible Server

Edit [samples/Zhengyan.QwenSharp.Server/appsettings.json](samples/Zhengyan.QwenSharp.Server/appsettings.json):

```json
{
  "Urls": "http://0.0.0.0:5000",
  "QwenSharp": {
    "ModelPath": "/data/models/Qwen2.5-Omni-7B",
    "ModelName": "",
    "Device": "cuda:0",
    "DType": "float16",
    "DeviceMap": null,
    "DisableTalker": false,
    "NoCudaCache": false,
    "Realtime": {
      "DisableMultiGpu": false
    }
  }
}
```

Recommended multi-GPU server template for stable Realtime:

```json
{
  "QwenSharp": {
    "ModelPath": "/data/models/Qwen2.5-Omni-7B",
    "Device": "cuda:0",
    "DType": "float16",
    "DeviceMap": "cuda:0,cuda:1",
    "Realtime": {
      "DisableMultiGpu": true
    }
  }
}
```

This keeps `/v1/chat/completions` and `/v1/responses` on the multi-GPU path, while `/v1/realtime` and `/v1/audio/speech` use a single-device Omni instance pinned to the first device in `DeviceMap`.

Then start the server:

```bash
dotnet run --project samples/Zhengyan.QwenSharp.Server
```

Command-line overrides are still supported when you want a one-off launch:

```bash
dotnet run --project samples/Zhengyan.QwenSharp.Server -- --model-path /data/models/Qwen2.5-Omni-7B --device cuda:0 --dtype float16
```

The server reads `config.json` and automatically detects whether the model is `qwen2_5_omni`.

- Omni models expose `/v1/realtime`, text Chat Completions, and text Responses. The same Omni service path also accepts multimodal inputs when the request contains image, audio, or video parts.
- Non-Omni models expose text Chat Completions and text Responses only. Realtime is not mapped for non-Omni models.

### Server Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/` | Server metadata |
| `GET` | `/v1/models` | Single loaded model entry |
| `POST` | `/v1/chat/completions` | OpenAI-style Chat Completions; text for all models, multimodal through Omni |
| `POST` | `/v1/responses` | OpenAI-style Responses; text for all models, multimodal through Omni |
| `GET` | `/v1/realtime` | WebSocket Realtime API; Omni only |

### Chat Completions Example

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [
      { "role": "user", "content": "Write a short introduction to QwenSharp." }
    ],
    "temperature": 0.2,
    "max_tokens": 256
  }'
```

### Multimodal Responses Example

Multimodal input is supported by the Omni service path.

```bash
curl http://localhost:5000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2_5_omni",
    "input": [
      {
        "role": "user",
        "content": [
          { "type": "input_text", "text": "What is in this image?" },
          { "type": "input_image", "image_url": "https://example.com/image.jpg" }
        ]
      }
    ],
    "max_output_tokens": 256
  }'
```

The adapter accepts common OpenAI-style content part forms, including `text`, `input_text`, `image_url`, `input_image`, audio input, and video input parts.

### Function Calling

The OpenAI contract layer accepts `tools`, `tool_choice`, `tool_calls`, and function-call shaped messages. For local models, function calling is model-driven: the service adds tool instructions to the prompt, parses tool-call JSON from the assistant output, and returns OpenAI-style tool call objects. It does not execute your functions inside the server.

```json
{
  "model": "qwen",
  "messages": [
    { "role": "user", "content": "What is the weather in Hangzhou?" }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": { "type": "string" }
          },
          "required": [ "city" ]
        }
      }
    }
  ]
}
```

Your client should execute the returned function call and send the tool result back in the next request.

## Run the Qwen2.5-Omni Browser Demo

The Omni web demo is the easiest way to test realtime voice and multimodal interaction from a browser.

Edit [samples/Zhengyan.QwenSharp.OmniWebDemo/appsettings.json](samples/Zhengyan.QwenSharp.OmniWebDemo/appsettings.json), especially:

```json
{
  "QwenSharp": {
    "ModelPath": "/data/models/Qwen2.5-Omni-7B",
    "Device": "cuda:0",
    "DType": "float16"
  }
}
```

Start it:

```bash
dotnet run --project samples/Zhengyan.QwenSharp.OmniWebDemo
```

Open:

```text
http://localhost:5000
```

Browser microphone capture requires a secure browser context. `http://localhost` works for local testing. For remote hosts, use HTTPS or browser settings that explicitly allow microphone access.

The demo can:

- open a Realtime WebSocket session
- capture microphone audio
- auto-submit a turn after silence
- stream assistant text
- play synthesized audio
- send text, image URL, and video inputs

## TTS Configuration

Speech output is produced by the configured C# TTS backend. The sample configs expect Sherpa-ONNX Matcha files under:

```text
models/
  tts/
    matcha-icefall-zh-en/
      model-steps-3.onnx
      tokens.txt
      vocos-16khz-univ.onnx
      lexicon.txt
      espeak-ng-data/
```

You can change the paths under `QwenSharp:Tts` in `appsettings.json`. The sample projects copy the repository `models/` directory to the output directory when needed, and the runtime also resolves paths from the current directory, output directory, content root, and repository root.

Before the first run, download the default TTS assets with one of the helper scripts:

```bash
sh scripts/download-tts-models.sh
```

```powershell
.\scripts\download-tts-models.ps1
```

The scripts download `matcha-icefall-zh-en.tar.bz2`, extract it to `models/tts`, then download `vocos-16khz-univ.onnx` into `models/tts/matcha-icefall-zh-en`.
On Windows, if the system `tar.exe` cannot extract `.tar.bz2` because `bzip2` is missing, the PowerShell script automatically falls back to a temporary .NET extractor.

## Device and Memory Notes

Common settings:

```json
{
  "QwenSharp": {
    "Device": "cuda:0",
    "DType": "float16",
    "DeviceMap": null,
    "NoCudaCache": false
  }
}
```

For multi-GPU Omni inference, set `DeviceMap`:

```json
{
  "QwenSharp": {
    "DeviceMap": "cuda:0,cuda:1",
    "DType": "float16"
  }
}
```

Recommended server template for stable Realtime and speech synthesis:

```json
{
  "QwenSharp": {
    "ModelPath": "/data/models/Qwen2.5-Omni-7B",
    "Device": "cuda:0",
    "DType": "float16",
    "DeviceMap": "cuda:0,cuda:1",
    "Realtime": {
      "DisableMultiGpu": true
    }
  }
}
```

If you want to keep regular HTTP inference on a multi-GPU Omni service but force Realtime and `/v1/audio/speech` back to a single GPU, enable:

```json
{
  "QwenSharp": {
    "DeviceMap": "cuda:0,cuda:1",
    "Realtime": {
      "DisableMultiGpu": true
    }
  }
}
```

When `Realtime.DisableMultiGpu` is `true`, `Zhengyan.QwenSharp.Server` loads an additional single-device Omni service pinned to the first device from `DeviceMap`. The regular `/v1/chat/completions` and `/v1/responses` endpoints still use the multi-GPU service, while `/v1/realtime` and `/v1/audio/speech` use the single-GPU service for better conversational stability. This improves Realtime reliability, but it also means the process keeps another copy of the Omni model in memory.

You can also pass `--device-map auto` to use all visible CUDA devices in order. If you set `CUDA_VISIBLE_DEVICES=2,3`, the process sees those cards as `cuda:0` and `cuda:1`.

If CUDA memory fragmentation is a problem, enable:

```bash
export QWENSHARP_NO_CUDA_CACHE=1
```

or set:

```json
{
  "QwenSharp": {
    "NoCudaCache": true
  }
}
```

## Linux Native Runtime Notes

On Linux, make sure the directory containing `libtorch.so` is in `LD_LIBRARY_PATH` before starting the app:

```bash
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
dotnet run --project samples/Zhengyan.QwenSharp.Server
```

The samples are config-first and can be started with plain `dotnet run`. The `scripts/` directory only contains optional TTS model download helpers.

## CLI Sample

```bash
dotnet run --project samples/Zhengyan.QwenSharp.Cli -- chat --model-path /data/models/Qwen2-0.5B-Instruct
dotnet run --project samples/Zhengyan.QwenSharp.Cli -- generate "Hello" --model-path /data/models/Qwen2-0.5B-Instruct
```

## Cross-Platform Policy

Project code uses .NET APIs for path handling, configuration, HTTP, WebSocket, and media processing glue. Platform-specific behavior is limited to native TorchSharp/LibTorch runtime loading and GPU driver availability.

The intended deployment targets are:

- Windows
- Linux
- macOS

## Security Notes

The sample servers do not implement authentication, rate limiting, or request isolation. Treat them as development or trusted-network samples. Add your own authentication and operational controls before exposing them publicly.

## License

Apache 2.0
