# Zhengyan.QwenSharp

中文 | [English](README.md)

Zhengyan.QwenSharp 是一个纯 C# 的 Qwen 系列模型推理项目。它基于 .NET 10 和 TorchSharp 加载本地 Qwen 权重，完成文本生成、多模态输入处理，并提供 OpenAI 兼容的 HTTP 和 Realtime API。项目不引入 Python 服务层。

项目由核心推理库、OpenAI 协议适配层，以及几个可运行的 sample 组成：CLI、本地浏览器 Omni Demo、通用 OpenAI 兼容 Server。

## 功能概览

- 从本地 `config.json`、tokenizer 文件、SafeTensors 或二进制权重加载 Qwen 模型。
- 支持 Qwen 因果语言模型的文本生成。
- 通过 C# Omni 适配器支持 Qwen2.5-Omni 的文本、语音、图像、视频理解。
- 支持浏览器麦克风采集和实时语音交互。
- 支持通过可配置的 C# TTS 后端返回语音。
- 提供 OpenAI 风格的 `/v1/chat/completions`、`/v1/responses`、`/v1/realtime` 接口。
- 支持解析 OpenAI 风格的多模态 message parts 和 function/tool call 请求。
- 支持 Windows、Linux、macOS，具体取决于 TorchSharp 和 LibTorch native runtime 的平台支持。

## 项目结构

```text
src/
  Zhengyan.QwenSharp/
    核心推理库。包含模型加载、tokenizer、generation 工具、Qwen 模型实现、
    vision 工具，以及 Qwen2.5-Omni 的音频/视觉处理。

  Zhengyan.QwenSharp.OpenAI/
    OpenAI 协议适配层。负责 HTTP/WebSocket 请求映射、OpenAI 请求/响应
    DTO，以及 Chat Completions、Responses、Realtime 协议处理。

samples/
  Zhengyan.QwenSharp.Cli/
    命令行 sample，用于本地生成和一些轻量工具流程。

  Zhengyan.QwenSharp.OmniWebDemo/
    Qwen2.5-Omni 浏览器 Demo，支持实时语音、图片和视频交互。

  Zhengyan.QwenSharp.Server/
    可配置的 OpenAI 兼容服务。可以加载普通 Qwen 文本模型，也可以加载
    Qwen2.5-Omni。模型为 Omni 时会自动启用 Realtime API。
```

## 支持的模型系列

| 模型系列 | 能力 | 状态 |
| --- | --- | --- |
| Qwen2 | 文本生成 | 已支持 |
| Qwen2-MoE | 文本生成 | 已支持 |
| Qwen3 | 文本生成 | 已支持 |
| Qwen3-MoE | 文本生成 | 已支持 |
| Qwen3.5 | 文本生成 | 已支持 |
| Qwen3.5-MoE | 文本生成 | 已支持 |
| Qwen2-VL | 视觉语言模型类 | 核心实现已提供 |
| Qwen2.5-VL | 视觉语言模型类 | 核心实现已提供 |
| Qwen3-VL | 视觉语言模型类 | 核心实现已提供 |
| Qwen2-Audio | 音频语言模型类 | 核心实现已提供 |
| Qwen2.5-Omni | 文本、语音、图像、视频、Realtime | 通过 Omni service path 支持 |
| Classification 和 QA heads | 序列分类、Token 分类、问答 | 已支持 |
| ColQwen2 | 检索模型 | 已支持 |

通用的 `Zhengyan.QwenSharp.Server` 在非 Omni 模型下目前提供文本 OpenAI 接口。多模态 OpenAI 输入当前主要接入在 Qwen2.5-Omni service path。

## 环境要求

- .NET 10 SDK
- TorchSharp 兼容的 native runtime
- 本地 Qwen 模型目录
- 大模型推荐使用 CUDA GPU
- 如果需要语音输出，需要可选的 Sherpa-ONNX TTS 模型文件

项目本身不需要 Python 应用进程。Linux 上仍然需要让 TorchSharp 能找到兼容的 LibTorch 或 PyTorch native libraries。

## 构建

```bash
dotnet build Zhengyan.QwenSharp.slnx
```

## 模型目录

`ModelPath` 指向包含模型配置、tokenizer 和权重文件的目录。

```text
Models/
  Qwen2.5-Omni-7B/
    config.json
    tokenizer.json
    tokenizer_config.json
    *.safetensors
    *.bin
```

模型文件可以从官方 Qwen 仓库获取：

- https://huggingface.co/Qwen
- https://github.com/QwenLM

## 启动 OpenAI 兼容 Server

修改 [samples/Zhengyan.QwenSharp.Server/appsettings.json](samples/Zhengyan.QwenSharp.Server/appsettings.json)：

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

推荐的多卡 Server 配置模板如下，适合“普通 HTTP 请求走多卡，Realtime 保持单卡稳定”的场景：

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

这样配置后，`/v1/chat/completions` 和 `/v1/responses` 继续走多卡路径，而 `/v1/realtime` 和 `/v1/audio/speech` 会切到绑定在 `DeviceMap` 第一张卡上的单卡 Omni 实例，实时对话更稳。

启动服务：

```bash
dotnet run --project samples/Zhengyan.QwenSharp.Server
```

如果只是临时覆盖配置，也可以继续使用命令行参数：

```bash
dotnet run --project samples/Zhengyan.QwenSharp.Server -- --model-path /data/models/Qwen2.5-Omni-7B --device cuda:0 --dtype float16
```

Server 会读取模型目录里的 `config.json`，自动判断是否为 `qwen2_5_omni`。

- Omni 模型会提供 `/v1/realtime`、文本 Chat Completions 和文本 Responses 接口。同一个 Omni service path 在请求包含图片、音频或视频 parts 时，也会处理多模态输入。
- 非 Omni 模型只提供文本 Chat Completions 和文本 Responses 接口，不会映射 Realtime API。

### Server 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 服务元信息 |
| `GET` | `/v1/models` | 当前加载的模型 |
| `POST` | `/v1/chat/completions` | OpenAI 风格 Chat Completions；所有模型支持文本，Omni 支持多模态 |
| `POST` | `/v1/responses` | OpenAI 风格 Responses；所有模型支持文本，Omni 支持多模态 |
| `GET` | `/v1/realtime` | WebSocket Realtime API；仅 Omni 模型启用 |

### Chat Completions 示例

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [
      { "role": "user", "content": "介绍一下 QwenSharp。" }
    ],
    "temperature": 0.2,
    "max_tokens": 256
  }'
```

### 多模态 Responses 示例

多模态输入由 Omni service path 支持。

```bash
curl http://localhost:5000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2_5_omni",
    "input": [
      {
        "role": "user",
        "content": [
          { "type": "input_text", "text": "这张图片里有什么？" },
          { "type": "input_image", "image_url": "https://example.com/image.jpg" }
        ]
      }
    ],
    "max_output_tokens": 256
  }'
```

协议层支持常见 OpenAI 风格 content part，包括 `text`、`input_text`、`image_url`、`input_image`、音频输入和视频输入。

### Function Calling

OpenAI 协议层支持接收 `tools`、`tool_choice`、`tool_calls` 和 function-call 形态的消息。对于本地模型，function calling 是模型驱动的：服务会把工具说明加入 prompt，尝试从助手输出中解析 tool-call JSON，并返回 OpenAI 风格的 tool call 对象。服务端不会直接执行你的函数。

```json
{
  "model": "qwen",
  "messages": [
    { "role": "user", "content": "杭州现在天气怎么样？" }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取城市当前天气。",
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

客户端应执行返回的函数调用，并在下一次请求中把工具结果发回模型。

## 启动 Qwen2.5-Omni 浏览器 Demo

OmniWebDemo 是测试实时语音和多模态交互最方便的入口。

修改 [samples/Zhengyan.QwenSharp.OmniWebDemo/appsettings.json](samples/Zhengyan.QwenSharp.OmniWebDemo/appsettings.json)，尤其是：

```json
{
  "QwenSharp": {
    "ModelPath": "/data/models/Qwen2.5-Omni-7B",
    "Device": "cuda:0",
    "DType": "float16"
  }
}
```

启动：

```bash
dotnet run --project samples/Zhengyan.QwenSharp.OmniWebDemo
```

打开：

```text
http://localhost:5000
```

浏览器麦克风采集需要安全上下文。本地测试使用 `http://localhost` 即可。远程访问时建议使用 HTTPS，或者在浏览器里显式允许该地址使用麦克风。

Demo 支持：

- 建立 Realtime WebSocket 会话
- 采集麦克风音频
- 静音后自动提交一轮输入
- 流式显示助手文本
- 播放合成语音
- 发送文本、图片 URL、视频输入

## TTS 配置

语音输出由配置的 C# TTS 后端生成。sample 默认期望 Sherpa-ONNX Matcha 文件位于：

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

你可以在 `appsettings.json` 的 `QwenSharp:Tts` 下修改路径。sample 项目会在需要时把仓库里的 `models/` 目录复制到输出目录；运行时也会从当前目录、输出目录、content root 和仓库根目录解析相对路径。

首次运行前，可以用下面任意一个脚本下载默认 TTS 资源：

```bash
sh scripts/download-tts-models.sh
```

```powershell
.\scripts\download-tts-models.ps1
```

脚本会下载 `matcha-icefall-zh-en.tar.bz2` 并解压到 `models/tts`，然后把 `vocos-16khz-univ.onnx` 下载到 `models/tts/matcha-icefall-zh-en`。
在 Windows 上，如果系统 `tar.exe` 因为缺少 `bzip2` 无法解压 `.tar.bz2`，PowerShell 脚本会自动切换到临时的 .NET 解压器。

## 设备和显存说明

常用配置：

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

Omni 多卡推理可以设置 `DeviceMap`：

```json
{
  "QwenSharp": {
    "DeviceMap": "cuda:0,cuda:1",
    "DType": "float16"
  }
}
```

推荐的 Server 配置模板如下，适合“普通 HTTP 请求走多卡，Realtime 和语音合成保持单卡稳定”的场景：

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

如果你希望普通 HTTP 请求继续走多卡，但把 Realtime 和 `/v1/audio/speech` 强制切回单卡，可以启用：

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

启用后，`Zhengyan.QwenSharp.Server` 会额外加载一份绑定到 `DeviceMap` 第一张卡的单卡 Omni 服务。`/v1/chat/completions` 和 `/v1/responses` 继续使用多卡服务，而 `/v1/realtime` 和 `/v1/audio/speech` 会切到单卡服务，以提高实时对话稳定性。代价是进程里会常驻另一份 Omni 模型显存/内存。

也可以通过 `--device-map auto` 使用当前进程可见的全部 CUDA 设备。如果设置了 `CUDA_VISIBLE_DEVICES=2,3`，进程内部会把这两张卡看作 `cuda:0` 和 `cuda:1`。

如果遇到 CUDA 显存碎片问题，可以启用：

```bash
export QWENSHARP_NO_CUDA_CACHE=1
```

或者在配置里设置：

```json
{
  "QwenSharp": {
    "NoCudaCache": true
  }
}
```

## Linux Native Runtime 说明

Linux 上启动应用前，需要确保包含 `libtorch.so` 的目录已经加入 `LD_LIBRARY_PATH`：

```bash
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
dotnet run --project samples/Zhengyan.QwenSharp.Server
```

sample 本身已经是配置优先，可以直接用 `dotnet run` 启动。`scripts/` 目录现在只保留可选的 TTS 模型下载脚本。

## CLI Sample

```bash
dotnet run --project samples/Zhengyan.QwenSharp.Cli -- chat --model-path /data/models/Qwen2-0.5B-Instruct
dotnet run --project samples/Zhengyan.QwenSharp.Cli -- generate "你好" --model-path /data/models/Qwen2-0.5B-Instruct
```

## 跨平台约定

项目代码使用 .NET API 处理路径、配置、HTTP、WebSocket 和媒体处理胶水逻辑。平台相关行为主要来自 TorchSharp/LibTorch native runtime 加载和 GPU 驱动。

目标平台：

- Windows
- Linux
- macOS

## 安全说明

sample server 没有实现认证、限流或请求隔离。它们适合作为开发环境或可信网络中的示例。公开部署前应自行添加认证和运维控制。

## License

Apache 2.0
