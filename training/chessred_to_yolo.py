#!/usr/bin/env python3
"""Convert ChessReD's annotated subset (ChessReD2K) to YOLO datasets.

ChessReD (Masouris & van Gemert, 2023) is 10,800 smartphone photos of real
games from many angles, including low side views. 2,078 of them have piece
bounding boxes and board corners, which is what this project's models train on:

- pieces/  detection dataset in the pieces model's 12 classes
- board/   segmentation dataset (one polygon per board, from its 4 corners)

Only those 2,078 photos are fetched: the 24.6 GB image archive is read with
HTTP range requests, so about 5 GB is downloaded instead. Images are resized
so the longer side is --max-side pixels.

Dataset licence: CC BY-NC-SA 4.0 (non-commercial, share-alike). Models
fine-tuned on it inherit those terms.

    python training/chessred_to_yolo.py --out datasets/chessred
"""

import argparse
import io
import json
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ANNOTATIONS_URL = "https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f/3cae6364-daca-4967-b426-1e4b68cdb64c"
IMAGES_URL = "https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f/6329e969-616e-48e3-b893-a0379d1c15ba"

# Class order of src/models/pieces-model (see rust/src/pieces.rs CLASSES).
PIECE_CLASSES = [
    "black-bishop", "black-king", "black-knight", "black-pawn", "black-queen", "black-rook",
    "white-bishop", "white-king", "white-knight", "white-pawn", "white-queen", "white-rook",
]


class HttpFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests, for zipfile."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise OSError(f"expected a partial response, got HTTP {r.status}")
                    data = r.read()
                if len(data) == end - self.pos + 1:
                    break
            except OSError as e:
                if attempt == 4:
                    raise
                print(f"retrying read ({e})", file=sys.stderr)
                time.sleep(2**attempt)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def load_annotations(path):
    if not path.exists():
        print(f"downloading annotations to {path}")
        urllib.request.urlretrieve(ANNOTATIONS_URL, path)
    return json.loads(path.read_text())


def open_archive(images):
    if images:
        return lambda: zipfile.ZipFile(images)
    size = HttpFile(IMAGES_URL).size
    print(f"reading images remotely from the {size / 1e9:.1f} GB archive (only the annotated ones)")
    return lambda: zipfile.ZipFile(io.BufferedReader(HttpFile(IMAGES_URL), buffer_size=1 << 20))


def board_polygon(corners):
    """The 4 corners in polygon order (by angle around their centre)."""
    import math

    pts = list(corners.values())
    cx = sum(p[0] for p in pts) / 4
    cy = sum(p[1] for p in pts) / 4
    return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("datasets/chessred"))
    ap.add_argument("--annotations", type=Path, help="annotations.json (downloaded if missing)")
    ap.add_argument("--images", type=Path, help="local images.zip; default reads the remote archive")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=0, help="convert only this many images per split (testing)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ann = load_annotations(args.annotations or args.out / "annotations.json")
    names = {c["id"]: c["name"] for c in ann["categories"]}
    images = {im["id"]: im for im in ann["images"]}
    corners = {c["image_id"]: c["corners"] for c in ann["annotations"]["corners"]}
    boxes = {}
    for p in ann["annotations"]["pieces"]:
        if "bbox" in p and names[p["category_id"]] in PIECE_CLASSES:
            boxes.setdefault(p["image_id"], []).append((PIECE_CLASSES.index(names[p["category_id"]]), p["bbox"]))

    jobs = []
    for split, info in ann["splits"]["chessred2k"].items():
        ids = [i for i in info["image_ids"] if i in corners]
        jobs += [(split, images[i]) for i in (ids[: args.limit] if args.limit else ids)]

    open_zip = open_archive(args.images)
    local = threading.local()

    def archive():
        # One open archive per thread: reading the remote index is slow.
        if not hasattr(local, "zip"):
            local.zip = open_zip()
        return local.zip

    members = {Path(n).name: n for n in archive().namelist()}

    def convert(job):
        split, im = job
        stem = Path(im["file_name"]).stem
        img = Image.open(io.BytesIO(archive().read(members[im["file_name"]]))).convert("RGB")
        w, h = img.size
        scale = min(1.0, args.max_side / max(w, h))
        if scale < 1:
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        for task in ("pieces", "board"):
            (args.out / task / "images" / split).mkdir(parents=True, exist_ok=True)
            (args.out / task / "labels" / split).mkdir(parents=True, exist_ok=True)
            img.save(args.out / task / "images" / split / f"{stem}.jpg", quality=92)
        piece_lines = [
            f"{cls} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
            for cls, (x, y, bw, bh) in boxes.get(im["id"], [])
        ]
        (args.out / "pieces" / "labels" / split / f"{stem}.txt").write_text("\n".join(piece_lines) + "\n")
        poly = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in board_polygon(corners[im["id"]]))
        (args.out / "board" / "labels" / split / f"{stem}.txt").write_text(f"0 {poly}\n")
        return split

    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for _ in pool.map(convert, jobs):
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"\r{done}/{len(jobs)} images", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    for task, names_yaml in (("pieces", PIECE_CLASSES), ("board", ["chess_board"])):
        root = (args.out / task).resolve()
        lines = [f"path: {root}", "train: images/train", "val: images/val", "test: images/test", "names:"]
        lines += [f"  {i}: {n}" for i, n in enumerate(names_yaml)]
        (args.out / f"{task}.yaml").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}/pieces.yaml and {args.out}/board.yaml")


if __name__ == "__main__":
    main()
