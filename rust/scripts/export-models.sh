#!/bin/sh
# Exports the YOLO models in src/models from PyTorch (.pt) to ONNX, which the
# Rust CLI and desktop app load, and installs ONNX Runtime next to them (the
# programs find it there without ORT_DYLIB_PATH). Run once from anywhere;
# needs Python 3.
# Packages go into a virtual environment (.venv at the repository root), so
# this also works with Homebrew's Python, which refuses global pip installs.
set -eu

cd "$(dirname "$0")/../.."
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
if [ "$(uname -s)" = Linux ]; then
  # Exporting doesn't need a GPU: the CPU-only PyTorch build is several GB smaller.
  # Falls back to the default build below if that index is unreachable.
  .venv/bin/python -m pip install --quiet --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu || true
fi
.venv/bin/python -m pip install --quiet --no-cache-dir "ultralytics==8.3.31" onnx onnxscript onnxruntime

for m in board pieces hand; do
  .venv/bin/yolo export model="src/models/$m-model.pt" format=onnx imgsz=640
done

ls -l src/models/*.onnx
