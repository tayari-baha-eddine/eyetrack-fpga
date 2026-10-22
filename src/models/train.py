import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.models.mlp import EyeMLP
from src.evaluation.metrics import error_px, summary

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed/swirski")  # <- adapte a Swirski
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
d = Path(args.data)
X = np.load(d / "images_16x16.npy").astype(np.float32) / 255.0
Y = np.load(d / "targets_xy.npy").astype(np.float32)
S = np.load(d / "splits.npy")


def part(name):
    m = S == name
    return torch.from_numpy(X[m]), torch.from_numpy(Y[m])


Xtr, Ytr = part("train")
Xva, Yva = part("val")
model = EyeMLP()
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()
Path("results").mkdir(exist_ok=True)
Path("artifacts").mkdir(exist_ok=True)

best, log = float("inf"), []
for ep in range(args.epochs):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), args.batch):
        idx = perm[i:i + args.batch]
        opt.zero_grad()
        loss = loss_fn(model(Xtr[idx]), Ytr[idx])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pv = model(Xva).numpy()
    s = summary(error_px(pv, Yva.numpy()))
    log.append({"epoch": ep, **s})
    if s["mean_px"] < best:
        best = s["mean_px"]
        torch.save(model.state_dict(), "artifacts/mlp_fp32.pt")
    if ep % 50 == 0:
        print(f"epoch {ep:3d} | val mean {s['mean_px']:.2f} px | best {best:.2f} px")

with open("results/train_log.json", "w") as f:
    json.dump(log, f)
print("Meilleure erreur moyenne en validation (px ROI) :", best)
