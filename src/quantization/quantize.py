import argparse
from pathlib import Path
import numpy as np
import torch
from src.models.mlp import EyeMLP


def mult_shift(ratio):
    # approxime ratio par M / 2**S avec M entre 2**14 et 2**15
    S = 1
    while ratio * 2 ** (S + 1) < 2 ** 15:
        S += 1
    assert S <= 40, "rapport trop petit"
    M = int(round(ratio * 2 ** S))
    assert 0 < M < 2 ** 16
    return M, S


def qw(W, scale):
    return np.clip(np.round(W / scale), -127, 127).astype(np.int64)


ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed/swirski")
ap.add_argument("--out", default="artifacts/qparams.npz")
ap.add_argument("--activation-percentile", type=float, default=99.9,
                help="percentile utilise pour calibrer les activations")
args = ap.parse_args()

model = EyeMLP()
model.load_state_dict(torch.load("artifacts/mlp_fp32.pt"))
W1, b1, W2, b2, W3, b3 = [p.detach().numpy().astype(np.float64)
                          for p in model.parameters()]

d = Path(args.data)
X = np.load(d / "images_16x16.npy").astype(np.float64) / 255.0
S = np.load(d / "splits.npy")
X = X[S == "train"].reshape(-1, 256)
a1 = np.maximum(X @ W1.T + b1, 0)
a2 = np.maximum(a1 @ W2.T + b2, 0)

# --- Calibration robuste : percentile au lieu de max ---
s_x = 1.0 / 255.0
s_w1, s_w2, s_w3 = [np.abs(W).max() / 127 for W in (W1, W2, W3)]
s_a1 = np.percentile(a1, args.activation_percentile) / 127
s_a2 = np.percentile(a2, args.activation_percentile) / 127

print(f"Calibration : s_a1={s_a1:.4f}  s_a2={s_a2:.4f}  "
      f"(percentile {args.activation_percentile})")
print(f"              a1.max()={a1.max():.2f}  a1.p99.9={np.percentile(a1, 99.9):.2f}")
print(f"              a2.max()={a2.max():.2f}  a2.p99.9={np.percentile(a2, 99.9):.2f}")

M1, S1 = mult_shift(s_x * s_w1 / s_a1)
M2, S2 = mult_shift(s_a1 * s_w2 / s_a2)
params = {
    "W1": qw(W1, s_w1),
    "b1": np.round(b1 / (s_x * s_w1)).astype(np.int64),
    "M1": M1, "S1": S1,
    "W2": qw(W2, s_w2),
    "b2": np.round(b2 / (s_a1 * s_w2)).astype(np.int64),
    "M2": M2, "S2": S2,
    "W3": qw(W3, s_w3),
    "b3": np.round(b3 / (s_a2 * s_w3)).astype(np.int64),
    "out_scale": s_a2 * s_w3,
}
Path(args.out).parent.mkdir(exist_ok=True)
np.savez(args.out, **{k: np.array(v) for k, v in params.items()})
print("M1, S1 =", M1, S1, "| M2, S2 =", M2, S2, "| out_scale =", params["out_scale"])
print("Fichier :", args.out)
