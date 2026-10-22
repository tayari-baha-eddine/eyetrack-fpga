import numpy as np
from src.quantization.quantized_reference import requant, forward_int

def test_requant_relu_and_saturation():
    acc = np.array([-1000, 0, 100, 10 ** 6], dtype=np.int64)
    assert requant(acc, 16384, 15).tolist() == [0, 0, 50, 127]

def test_requant_rounding():
    assert int(requant(np.array([145], dtype=np.int64), 16384, 15)[0]) == 73

def test_forward_int_hand_computed():
    qp = {"W1": np.ones((16, 256), dtype=np.int64), "b1": np.zeros(16, dtype=np.int64), "M1": 16384, "S1": 15,
          "W2": np.ones((8, 16), dtype=np.int64), "b2": np.zeros(8, dtype=np.int64), "M2": 16384, "S2": 15,
          "W3": np.ones((2, 8), dtype=np.int64), "b3": np.zeros(2, dtype=np.int64)}
    out = forward_int(np.ones((1, 256), dtype=np.int64), qp)
    assert out.tolist() == [[1016, 1016]]
