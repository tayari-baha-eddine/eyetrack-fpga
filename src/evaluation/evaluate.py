import argparse
from pathlib import Path
import numpy as np
import torch
from src.models.mlp import EyeMLP
from src.evaluation import baseline
from src.evaluation.metrics import error_px, summary
from src.quantization.quantized_reference import load_params, forward_int

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed/swirski")  # <- adapte a Swirski
ap.add_argument("--split", default="val",
                help="val pendant le dev, test une seule fois a la fin")
args = ap.parse_args()

d = Path(args.data)
X = np.load(d / "images_16x16.npy")
Y = np.load(d / "targets_xy.npy")
S = np.load(d / "splits.npy")
Xs, Ys = X[S == args.split], Y[S == args.split]
if len(Xs) == 0:
    raise SystemExit(f"Aucun exemple dans la partition {args.split}")

model = EyeMLP()
model.load_state_dict(torch.load("artifacts/mlp_fp32.pt"))
model.eval()
with torch.no_grad():
    p_fp = model(torch.from_numpy(Xs.astype(np.float32) / 255.0)).numpy()

qp = load_params()
p_int = forward_int(Xs.reshape(len(Xs), 256), qp) * float(qp["out_scale"])

thr = baseline.fit_threshold(X[S == "train"], Y[S == "train"])
p_b = baseline.predict(Xs, thr)
fail = np.isnan(p_b[:, 0])
p_b[fail] = 0.5

rows = [
    (f"Baseline (seuil {thr}, echecs {fail.mean():.1%})", p_b),
    ("MLP flottant", p_fp),
    ("MLP entier INT8/INT32", p_int),
]
lines = [
    f"Partition : {args.split} ({len(Xs)} images). Erreurs en pixels de la ROI 400x400.",
    "",
    "| Modele | moyenne | mediane | p95 | <1 % | <2 % | <5 % |",
    "| --- | --- | --- | --- | --- | --- | --- |",
]
for name, p in rows:
    s = summary(error_px(p, Ys))
    lines.append(
        f"| {name} | {s['mean_px']:.2f} | {s['median_px']:.2f} | {s['p95_px']:.2f} "
        f"| {s['within_1pct']:.1%} | {s['within_2pct']:.1%} | {s['within_5pct']:.1%} |"
    )
q = error_px(p_fp, p_int)
lines += ["", f"Ecart flottant / entier : moyenne {q.mean():.3f} px, maximum {q.max():.3f} px"]

text = "\n".join(lines)
print(text)
Path("artifacts").mkdir(exist_ok=True)
Path(f"artifacts/results_{args.split}.md").write_text(text + "\n")
