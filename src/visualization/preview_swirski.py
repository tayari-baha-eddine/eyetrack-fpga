"""Show a few random samples with the annotated center overlaid."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed/swirski")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = Path(args.processed)
    imgs = np.load(d / "images_16x16.npy")
    targets = np.load(d / "targets_xy.npy")
    splits = np.load(d / "splits.npy")
    meta = json.loads((d / "sample_ids.json").read_text())
    config = json.loads((d / "preprocessing_config.json").read_text())
    roi = tuple(config["roi"])
    roi_w = config["roi_width"]
    block = roi_w / 16

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(imgs), min(args.n, len(imgs)), replace=False)

    fig, axes = plt.subplots(len(idx), 3, figsize=(10, 3.4 * len(idx)))
    axes = np.atleast_2d(axes)
    for row, k in enumerate(idx):
        m = meta[k]
        im = Image.open(m["image"]).convert("L")
        crop = im.crop(roi)
        xc = targets[k, 0] * (roi_w - 1)
        yc = targets[k, 1] * (roi_w - 1)

        a = axes[row]
        a[0].imshow(im, cmap="gray")
        a[0].add_patch(Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                                 roi[3] - roi[1], fill=False, edgecolor="red"))
        a[0].plot(m["xc_px"], m["yc_px"], "r+", markersize=14)
        a[0].set_title(f"image ({splits[k]})")

        a[1].imshow(crop, cmap="gray")
        a[1].plot(xc, yc, "r+", markersize=14)
        a[1].set_title(f"ROI {roi_w}px, centre ({xc:.1f}, {yc:.1f})")

        a[2].imshow(imgs[k], cmap="gray", extent=(0, 16, 16, 0))
        a[2].plot(xc / block, yc / block, "r+", markersize=14)
        a[2].set_title("16x16 + croix")

        for ax in a:
            ax.axis("off")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    out = Path(f"reports/figures/preview_swirski_seed{args.seed}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"Figure : {out}")


if __name__ == "__main__":
    main()
    
