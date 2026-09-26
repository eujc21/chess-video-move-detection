# chess-video-moves (Rust port)

A Rust port of the Python pipeline in this repository. It reads a video of a chess game and writes the moves in algebraic notation to a CSV file (`row_id,output`), using the same three YOLO models.

## Requirements

- **Rust** 1.88 or newer.
- **ffmpeg and ffprobe** on `PATH`. They decode the video.
- **ONNX Runtime** 1.17 or newer as a shared library. It is loaded at runtime:
  - Set `ORT_DYLIB_PATH` to point at the library. The `onnxruntime` pip package ships one at `site-packages/onnxruntime/capi/libonnxruntime.so.*`.
  - Or put `libonnxruntime.so` / `onnxruntime.dll` on the library path.
- **ONNX exports of the models.** The `.pt` files can't be loaded outside PyTorch, so export them once:

  ```bash
  pip install ultralytics==8.3.31 onnx onnxscript
  for m in board pieces hand; do
    yolo export model=src/models/$m-model.pt format=onnx imgsz=640
  done   # writes src/models/*-model.onnx
  ```

## Usage

From the repository root:

```bash
cargo build --release --manifest-path rust/Cargo.toml
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
  rust/target/release/chess-video-moves src/inputs/*.mp4 -o result.csv
```

Useful options (`--help` lists them all):

| Option | Default | Meaning |
| --- | --- | --- |
| `--interval` | `1` | Seconds between analysed frames. |
| `--hand-checks-per-second` | `5` | How often the hand model runs. Set it to the video's fps to match the Python behaviour, which checks every frame. |
| `--hand-confidence` | `0.65` | Minimum confidence for a hand detection. |
| `--stable-samples` | `2` | How many consecutive samples must agree before a position is used. |
| `--max-plies` | `2` | Most moves inferred between two accepted positions, for moves hidden while a hand was over the board. |
| `--max-residual` | `1.5` | Largest board mismatch allowed after a matched move. |
| `--board-corners "x,y x,y x,y x,y"` | none | Fixed board corners (TL TR BR BL). Use it for a fixed camera or when the board model fails. |
| `--check-marks` | off | Append `+` / `#` to moves. |
| `--naive` | off | Use the original frame-diff move detection instead of legal-move matching. |
| `--threads` | `0` | ONNX Runtime intra-op threads. `0` means the runtime default. |

Build with `--features cuda` or `--features tensorrt` to use the GPU. This needs a matching ONNX Runtime build.

## What changed compared to the Python version

**Speed**

- **Hand model runs less often.** The Python code runs the hand model on every decoded frame. Here it runs about `--hand-checks-per-second` times per second. A sample is processed only after its ±1 s window has been checked, so frames are still banned around hands.
- **Only needed frames are decoded.** ffmpeg's `select` filter decodes only sample frames and hand-check frames to RGB.
- **Pieces model runs once per sample.** The Python code runs it twice on the first frame to find the rotation. Here the rotation only remaps squares, so the same detections are reused.
- **O(1) square lookup.** A homography replaces the 64 Shapely `contains` tests per piece. Box/square overlaps use a small Sutherland–Hodgman clip with an early bounding-box rejection.
- **Offset vector is a running mean.** The Python list grows every frame and is re-averaged each time.
- **Models load once.** They are shared across all videos instead of reloaded per video.

**Accuracy**

- **Legal-move matching.** The tracker keeps a real chess position (via [`shakmaty`](https://crates.io/crates/shakmaty)). For each new observation it searches legal moves, up to two plies, for the resulting board that best matches the detections. This:
  - rejects detector noise that would otherwise become bogus moves,
  - recovers a move missed while a hand covered the board,
  - handles castling, en passant, promotion and SAN disambiguation.

  The side to move is inferred from the first move, which replaces the `first_move_black` heuristic. If the first position isn't a legal chess position, the tool falls back to the original diff logic.
- **Stability filter.** A position is accepted only after `--stable-samples` consecutive agreeing observations.
- **More robust rotation detection.**
  - Square colour is judged from the average brightness of *all* empty light versus dark squares, not a single pair.
  - The white side is judged from the average position of white versus black pieces.
  - The four rotations are proper rotations. The Python `position_to_tile` mapping mirrors the board for rotations 1 and 3.

## Tests

```bash
cargo test --manifest-path rust/Cargo.toml
```
