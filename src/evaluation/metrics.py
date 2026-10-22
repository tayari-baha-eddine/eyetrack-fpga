import numpy as np

ROI_W = 400
SCALE = 399.0  # ROI 400x400 -> coordonnees 0..399


def error_px(pred, target):
    # pred et target : (N, 2) normalises ; erreur euclidienne en pixels de la ROI
    return np.sqrt(((pred - target) ** 2).sum(axis=1)) * SCALE


def summary(err):
    err = np.asarray(err, dtype=np.float64)
    out = {"n": int(err.size),
           "mean_px": float(err.mean()),
           "median_px": float(np.median(err)),
           "p95_px": float(np.percentile(err, 95))}
    for pct in (1, 2, 5):
        out[f"within_{pct}pct"] = float((err / ROI_W <= pct / 100).mean())
    return out
