import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from src.models.mlp import EyeMLP
from src.evaluation import baseline
from src.evaluation.metrics import error_px
from src.quantization.quantized_reference import load_params, forward_int

SPLIT = "val"
d = Path("data/processed/swirski")
X = np.load(d / "images_16x16.npy")
Y = np.load(d / "targets_xy.npy")
S = np.load(d / "splits.npy")
meta = json.loads((d / "sample_ids.json").read_text())
config = json.loads((d / "preprocessing_config.json").read_text())
roi = tuple(config["roi"])

idx = np.where(S == SPLIT)[0]
Xs, Ys = X[idx], Y[idx]

qp = load_params()
p_int = forward_int(Xs.reshape(len(Xs), 256), qp) * float(qp["out_scale"])

model = EyeMLP()
model.load_state_dict(torch.load("artifacts/mlp_fp32.pt"))
model.eval()
with torch.no_grad():
    p_fp = model(torch.from_numpy(Xs.astype(np.float32) / 255.0)).numpy()

thr = baseline.fit_threshold(X[S == "train"], Y[S == "train"])
p_b = baseline.predict(Xs, thr)
p_b[np.isnan(p_b[:, 0])] = 0.5

err_int = error_px(p_int, Ys)

def draw(pick, path, title):
    fig, axes = plt.subplots(1, len(pick), figsize=(3 * len(pick), 3.6))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, pick):
        im = Image.open(meta[idx[k]]["image"]).convert("L").crop(roi)
        ax.imshow(im, cmap="gray")
        ax.plot(Ys[k, 0] * 399, Ys[k, 1] * 399, "g+", markersize=16, label="annoté")
        ax.plot(p_int[k, 0] * 399, p_int[k, 1] * 399, "rx", markersize=12, label="MLP entier")
        ax.set_title(f"erreur {err_int[k]:.1f} px")
        ax.axis("off")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)

Path("reports/figures").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

draw(rng.choice(len(idx), 6, replace=False), "reports/figures/fig1_random.png", "Exemples tirés au hasard")
draw(np.argsort(err_int)[-6:], "reports/figures/fig1_worst.png", "Les 6 pires cas")

plt.figure(figsize=(6, 4))
for name, p in (("Baseline", p_b), ("MLP flottant", p_fp), ("MLP entier", p_int)):
    e = np.sort(error_px(p, Ys))
    plt.plot(e, np.arange(1, len(e) + 1) / len(e), label=name)
for pct in (1, 2, 5):
    plt.axvline(4 * pct, color="gray", linestyle=":")
plt.xlim(0, 60)
plt.xlabel("erreur (px de la ROI 400x400)")
plt.ylabel("fraction des images")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/fig2_error_cdf.png", dpi=120)
print("Figures ecrites dans reports/figures/")
