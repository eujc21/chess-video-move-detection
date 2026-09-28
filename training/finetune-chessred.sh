#!/bin/sh
# Fine-tunes the pieces and board models on ChessReD's annotated photos
# (many camera angles, including low side views), compares them with the
# current models on ChessReD's held-out test photos, and exports ONNX files
# to src/models/chessred/ for the Rust CLI and desktop app.
#
# Needs Python 3; uses Apple's GPU (MPS) on Apple Silicon, CUDA if present,
# otherwise the CPU (slow). Extra arguments are passed to both `yolo train`
# runs, e.g. `training/finetune-chessred.sh epochs=10`.
#
# ChessReD is CC BY-NC-SA 4.0, so the fine-tuned models are for
# non-commercial use and must be shared under the same terms.
set -eu

cd "$(dirname "$0")/.."
data=datasets/chessred
out=src/models/chessred

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
# ultralytics 8.3.31 calls numpy.trapz, which numpy 2.4 removed.
.venv/bin/python -m pip install --quiet "ultralytics==8.3.31" "numpy<2.4" onnx onnxscript onnxruntime pillow

if [ ! -f "$data/pieces.yaml" ]; then
  .venv/bin/python training/chessred_to_yolo.py --out "$data"
fi

device=$(.venv/bin/python -c "import torch; print('mps' if torch.backends.mps.is_available() else 0 if torch.cuda.is_available() else 'cpu')")
echo "training on: $device"

for m in pieces board; do
  echo "== $m model: current, on ChessReD test photos"
  .venv/bin/yolo val model="src/models/$m-model.pt" data="$data/$m.yaml" split=test device="$device" \
    project=runs/chessred name="$m-before" exist_ok=True

  echo "== $m model: fine-tuning"
  .venv/bin/yolo train model="src/models/$m-model.pt" data="$data/$m.yaml" imgsz=640 epochs=50 patience=15 \
    device="$device" project=runs/chessred name="$m" exist_ok=True "$@"

  echo "== $m model: fine-tuned, on ChessReD test photos"
  .venv/bin/yolo val model="runs/chessred/$m/weights/best.pt" data="$data/$m.yaml" split=test device="$device" \
    project=runs/chessred name="$m-after" exist_ok=True

  .venv/bin/yolo export model="runs/chessred/$m/weights/best.pt" format=onnx imgsz=640
  mkdir -p "$out"
  cp "runs/chessred/$m/weights/best.onnx" "$out/$m-model.onnx"
done

# The hand model isn't retrained; the programs expect all three in one folder.
if [ ! -f src/models/hand-model.onnx ]; then
  .venv/bin/yolo export model=src/models/hand-model.pt format=onnx imgsz=640
fi
cp src/models/hand-model.onnx "$out/hand-model.onnx"
echo
echo "Done. Compare mAP50-95 in the 'before' and 'after' results above, then try them:"
echo "  rust/target/release/chess-video-gui --models $out"
