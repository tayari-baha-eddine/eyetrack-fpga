"""Prepare Swirski et al. 2012 for the EyeTrack-FPGA pipeline.

Split strategy (2 participants only, no strict participant-disjoint val):
  - p1-left + p1-right : train + val (every 5th annotation goes to val)
  - p2-left + p2-right : test (held-out participant 2)

ROI is a fixed square window, computed from participant 1 annotations
with a margin, then applied identically to all splits.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_ann_line(line):
    frame_str, params_str = line.split("|")
    frame = int(frame_str.strip())
    parts = params_str.split()
    xc, yc = float(parts[0]), float(parts[1])
    a, b, angle = float(parts[2]), float(parts[3]), float(parts[4])
    return frame, xc, yc, a, b, angle


def load_annotations(ann_path):
    ann = {}
    for line in ann_path.read_text().splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        try:
            frame, xc, yc, a, b, angle = parse_ann_line(line)
            ann[frame] = (xc, yc, a, b, angle)
        except (ValueError, IndexError):
            continue
    return ann


def compute_roi(annotation_dirs, target_size=400, margin=30):
    xs, ys = [], []
    for d in annotation_dirs:
        ann = load_annotations(d / "pupil-ellipses.txt")
        for (xc, yc, _, _, _) in ann.values():
            xs.append(xc)
            ys.append(yc)
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    half = target_size // 2
    x0 = max(0, int(round(cx - half)))
    y0 = max(0, int(round(cy - half)))
    return (x0, y0, x0 + target_size, y0 + target_size)


def split_frames(sorted_frames, every_nth_for_val=5):
    train, val = [], []
    for i, f in enumerate(sorted_frames):
        (val if i % every_nth_for_val == 0 else train).append(f)
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/swirski")
    ap.add_argument("--out", default="data/processed/swirski")
    ap.add_argument("--roi-size", type=int, default=400)
    ap.add_argument("--margin", type=int, default=30)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    folders = sorted(
        d for d in root.iterdir() if d.is_dir() and (d / "frames").exists()
    )
    if not folders:
        raise SystemExit(f"Aucun dossier frames/ dans {root}")

    train_dirs = [d for d in folders if d.name.startswith("p1")]
    test_dirs = [d for d in folders if d.name.startswith("p2")]
    if not train_dirs or not test_dirs:
        raise SystemExit(
            f"Attendu p1-* et p2-*, trouve: {[d.name for d in folders]}"
        )

    roi = compute_roi(train_dirs, args.roi_size, args.margin)
    x0, y0, x1, y1 = roi
    roi_w = x1 - x0
    print(f"Dossiers : {[d.name for d in folders]}")
    print(f"ROI (calculee sur p1) : {roi}")

    imgs, targets, splits, meta = [], [], [], []
    skipped = {"missing_img": 0, "out_of_roi": 0, "bad_size": 0}

    for folder in folders:
        frames_dir = folder / "frames"
        ann_path = folder / "pupil-ellipses.txt"
        if not ann_path.exists():
            print(f"  ! pas d'annotations dans {folder}")
            continue

        annotations = load_annotations(ann_path)
        sorted_frames = sorted(annotations.keys())

        if folder.name.startswith("p1"):
            train_f, val_f = split_frames(sorted_frames)
            split_of = {f: "train" for f in train_f}
            split_of.update({f: "val" for f in val_f})
        else:
            split_of = {f: "test" for f in sorted_frames}

        for frame_num in sorted_frames:
            img_path = frames_dir / f"{frame_num}-eye.png"
            if not img_path.exists():
                skipped["missing_img"] += 1
                continue
            xc, yc, a, b, angle = annotations[frame_num]
            if not (x0 <= xc < x1 and y0 <= yc < y1):
                skipped["out_of_roi"] += 1
                continue
            with Image.open(img_path) as im:
                if im.size != (620, 460):
                    skipped["bad_size"] += 1
                    continue
                crop = im.convert("L").crop(roi)
                small = np.asarray(
                    crop.resize((16, 16), Image.Resampling.BOX), dtype=np.uint8
                )
            imgs.append(small)
            targets.append(
                [(xc - x0) / (roi_w - 1), (yc - y0) / (roi_w - 1)]
            )
            splits.append(split_of[frame_num])
            meta.append(
                {
                    "image": str(img_path),
                    "folder": folder.name,
                    "frame": frame_num,
                    "xc_px": xc,
                    "yc_px": yc,
                    "semi_major": a,
                    "semi_minor": b,
                    "angle": angle,
                }
            )

    imgs = np.stack(imgs)
    targets = np.array(targets, dtype=np.float32)
    splits = np.array(splits)
    np.save(out / "images_16x16.npy", imgs)
    np.save(out / "targets_xy.npy", targets)
    np.save(out / "splits.npy", splits)
    (out / "sample_ids.json").write_text(json.dumps(meta))
    (out / "preprocessing_config.json").write_text(
        json.dumps(
            {
                "dataset": "Swirski et al. 2012 (ETRA)",
                "roi": list(roi),
                "roi_width": roi_w,
                "resize": [16, 16],
                "resample": "BOX",
                "target_norm": f"coordinate_px / {roi_w - 1}",
                "input_norm": "pixel / 255",
                "roi_computed_from": "participant 1 (train+val)",
                "split_strategy": "p1: train (4/5) + val (1/5), p2: test",
                "skipped": skipped,
            },
            indent=2,
        )
    )

    print(f"\nExemples conserves : {len(imgs)} | ecartes : {skipped}")
    for s in sorted(set(splits)):
        print(f"  {s:5s} : {(splits == s).sum()}")


if __name__ == "__main__":
    main()
