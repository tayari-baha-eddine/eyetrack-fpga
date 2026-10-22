import numpy as np


def predict(images, threshold):
    # images : (N, 16, 16) uint8 ; retourne (N, 2) normalise, NaN si echec
    preds = np.full((images.shape[0], 2), np.nan)
    gy, gx = np.mgrid[0:16, 0:16]
    for k in range(images.shape[0]):
        m = images[k] < threshold
        if m.any():
            cx, cy = gx[m].mean(), gy[m].mean()
            # chaque cellule du 16x16 couvre 25 pixels de la ROI (400/16)
            # centre de la cellule i : 25*i + 12
            preds[k] = [(25 * cx + 12) / 399, (25 * cy + 12) / 399]
    return preds


def fit_threshold(images, targets, candidates=range(5, 200, 5)):
    best = None
    for t in candidates:
        p = predict(images, t)
        ok = ~np.isnan(p[:, 0])
        if ok.mean() < 0.9:
            continue
        err = np.sqrt(((p[ok] - targets[ok]) ** 2).sum(axis=1)) * 399
        if best is None or err.mean() < best[1]:
            best = (t, err.mean())
    return best[0]


if __name__ == "__main__":
    from pathlib import Path
    from src.evaluation.metrics import error_px, summary

    d = Path("data/processed/swirski")
    X = np.load(d / "images_16x16.npy")
    Y = np.load(d / "targets_xy.npy")
    S = np.load(d / "splits.npy")

    t = fit_threshold(X[S == "train"], Y[S == "train"])
    p = predict(X[S == "val"], t)
    fail = np.isnan(p[:, 0])
    p[fail] = 0.5
    print("seuil", t, "| echecs", float(fail.mean()))
    print(summary(error_px(p, Y[S == "val"])))
