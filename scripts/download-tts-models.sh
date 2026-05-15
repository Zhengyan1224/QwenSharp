#!/usr/bin/env sh
set -eu

MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-en.tar.bz2"
VOCOS_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-16khz-univ.onnx"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
DEST_ROOT="$REPO_ROOT/models/tts"
MODEL_DIR="$DEST_ROOT/matcha-icefall-zh-en"
TMP_DIR="${TMPDIR:-/tmp}/qwensharp-tts-$$"
ARCHIVE_PATH="$TMP_DIR/matcha-icefall-zh-en.tar.bz2"
VOCOS_PATH="$MODEL_DIR/vocos-16khz-univ.onnx"

download_file() {
  url="$1"
  output="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --connect-timeout 20 -o "$output" "$url"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
    return
  fi

  echo "curl or wget is required to download model files." >&2
  exit 1
}

has_core_model_files() {
  [ -f "$MODEL_DIR/model-steps-3.onnx" ] \
    && [ -f "$MODEL_DIR/tokens.txt" ] \
    && [ -f "$MODEL_DIR/lexicon.txt" ] \
    && [ -d "$MODEL_DIR/espeak-ng-data" ]
}

if ! command -v tar >/dev/null 2>&1; then
  echo "tar is required to extract matcha-icefall-zh-en.tar.bz2." >&2
  exit 1
fi

mkdir -p "$DEST_ROOT" "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT INT TERM

if has_core_model_files; then
  echo "TTS model files already exist: $MODEL_DIR"
else
  echo "Downloading matcha-icefall-zh-en..."
  download_file "$MODEL_URL" "$ARCHIVE_PATH"

  echo "Extracting to $DEST_ROOT..."
  tar -xjf "$ARCHIVE_PATH" -C "$DEST_ROOT"
fi

mkdir -p "$MODEL_DIR"
if [ -f "$VOCOS_PATH" ]; then
  echo "Vocoder already exists: $VOCOS_PATH"
else
  echo "Downloading vocos-16khz-univ.onnx..."
  download_file "$VOCOS_URL" "$VOCOS_PATH"
fi

for required in \
  "$MODEL_DIR/model-steps-3.onnx" \
  "$MODEL_DIR/tokens.txt" \
  "$MODEL_DIR/lexicon.txt" \
  "$MODEL_DIR/espeak-ng-data" \
  "$MODEL_DIR/vocos-16khz-univ.onnx"
do
  if [ ! -e "$required" ]; then
    echo "Missing required TTS file after installation: $required" >&2
    exit 1
  fi
done

echo "TTS model is ready: $MODEL_DIR"
