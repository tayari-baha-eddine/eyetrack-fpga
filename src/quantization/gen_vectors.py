import argparse
from pathlib import Path
import numpy as np
from src.quantization.quantized_reference import load_params, forward_int

ap = argparse.ArgumentParser()
ap.add_argument("--real", type=int, default=0)
ap.add_argument("--data", default="data/processed/swirski")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

qp = load_params()
rng = np.random.default_rng(args.seed)
vecs = [np.zeros(256, dtype=np.int64), np.full(256, 255, dtype=np.int64)]
vecs.append(((np.indices((16, 16)).sum(axis=0) % 2) * 255).reshape(256))
vecs.append(np.round(np.linspace(0, 255, 256)).astype(np.int64))
for i in range(16):
    # entrees qui maximisent puis minimisent la somme du neurone i de la couche 1
    vecs.append(np.where(qp["W1"][i] > 0, 255, 0))
    vecs.append(np.where(qp["W1"][i] < 0, 255, 0))
vecs += [np.where(rng.random(256) < 0.5, 0, 255) for _ in range(20)]
vecs += [rng.integers(0, 256, 256) for _ in range(50)]


def write(path, V):
    V = np.array(V, dtype=np.int64)
    acc3, a1, a2 = forward_int(V, qp, full=True)
    lines = [" ".join(str(int(v)) for v in list(p) + list(o)) for p, o in zip(V, acc3)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(chr(10).join(lines) + chr(10))
    print(path, ":", len(V), "vecteurs | satures a1 :", int((a1 == 127).any(axis=1).sum()),
          "| satures a2 :", int((a2 == 127).any(axis=1).sum()),
          "| a1 nuls partout :", int((a1 == 0).all(axis=1).sum()))


write("tests/vectors/synthetic.txt", vecs)
if args.real:
    d = Path(args.data)
    X = np.load(d / "images_16x16.npy")
    S = np.load(d / "splits.npy")
    write("tests/vectors/real.txt", X[S == "val"][:args.real].reshape(-1, 256))
