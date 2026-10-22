import numpy as np


def load_params(path="artifacts/qparams.npz"):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def requant(acc, M, S):
    q = (acc * int(M) + (1 << (int(S) - 1))) >> int(S)
    return np.clip(q, 0, 127)


def forward_int(pix, qp, full=False):
    # pix : (N, 256) entiers 0..255 ; retourne l'accumulateur de la couche 3 (N, 2)
    x = np.asarray(pix, dtype=np.int64)
    acc1 = x @ qp["W1"].T + qp["b1"]
    a1 = requant(acc1, qp["M1"], qp["S1"])
    acc2 = a1 @ qp["W2"].T + qp["b2"]
    a2 = requant(acc2, qp["M2"], qp["S2"])
    acc3 = a2 @ qp["W3"].T + qp["b3"]
    worst = max(np.abs(acc1).max(), np.abs(acc2).max(), np.abs(acc3).max())
    assert worst < 2 ** 31, "debordement INT32"
    return (acc3, a1, a2) if full else acc3
