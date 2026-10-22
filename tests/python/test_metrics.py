import numpy as np
from src.evaluation.metrics import error_px, summary

def test_error_px_known_values():
    a = np.array([[0.5, 0.5]])
    assert error_px(a, a)[0] == 0
    b = np.array([[0.5 + 3 / 399, 0.5 + 4 / 399]])
    assert abs(error_px(b, a)[0] - 5.0) < 1e-9

def test_summary_thresholds():
    s = summary(np.array([1.0, 3.0, 10.0, 30.0]))
    assert s["mean_px"] == 11.0
    assert s["median_px"] == 6.5
    assert s["within_1pct"] == 0.5   # seuil 4 px
    assert s["within_5pct"] == 0.75  # seuil 20 px
