# Fine-tuning on ChessReD (side and low-angle views)

The shipped models were trained mostly on views from above. [ChessReD](https://github.com/tmasouris/end-to-end-chess-recognition) (Masouris & van Gemert, 2023) has 10,800 smartphone photos of real games from many angles, including low side views. 2,078 of those photos (the ChessReD2K subset) have piece bounding boxes and board corners, which is what these scripts train on.

On ChessReD's photos, the current board model often outlines the table instead of the board. The pipeline builds the square grid from that outline, so it can't place pieces on those photos at all. Fine-tuning mainly fixes this.

## Run it (on your Mac)

```bash
training/finetune-chessred.sh
```

The script:

1. **Converts the data.** `chessred_to_yolo.py` writes YOLO datasets to `datasets/chessred/`: piece boxes in this project's 12 classes, and board outlines from the 4 corners. It downloads only the ~2,000 annotated photos out of the 24.6 GB archive, using HTTP range requests, and shrinks them to 1280 px.
2. **Measures the current models** on ChessReD's test photos.
3. **Fine-tunes** the pieces and board models, starting from `src/models/*.pt`. It uses the Apple GPU (MPS) on Apple Silicon, CUDA if present, otherwise the CPU.
4. **Measures again** on the same test photos, so you can compare `mAP50-95` before and after.
5. **Exports ONNX** to `src/models/chessred/`, next to the hand model.

Extra arguments go to `yolo train`, for example `training/finetune-chessred.sh epochs=10 batch=8`.

Try the result:

```bash
rust/target/release/chess-video-gui --models src/models/chessred
```

The original models in `src/models/` are left unchanged.

## Licence

ChessReD is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Models fine-tuned on it are for non-commercial use, and must be shared under the same terms with credit to the dataset. That's why they go into a separate folder, which git ignores.
