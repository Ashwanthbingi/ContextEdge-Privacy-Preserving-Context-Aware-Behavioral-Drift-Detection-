from __future__ import annotations

import numpy as np
import onnxruntime as ort


class ONNXEngine:
    def __init__(self, model: bytes | str):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])

    def reconstruct(self, sequence) -> np.ndarray:
        array = np.asarray(sequence, dtype=np.float32)
        if array.ndim == 2:
            array = array[None]
        if not np.isfinite(array).all():
            raise ValueError("ONNX input must be finite")
        return self.session.run(["reconstruction"], {"features": array})[0]

    def scores(self, sequences) -> np.ndarray:
        return np.array([float(np.mean((self.reconstruct(seq)[0] - seq) ** 2)) for seq in sequences])


def verify_parity(model, engine: ONNXEngine, sequences, atol=1e-5, rtol=1e-4) -> dict:
    import torch

    errors = []
    for seq in sequences[:12]:
        with torch.no_grad():
            reference = model(torch.as_tensor(seq[None], dtype=torch.float32)).numpy()
        actual = engine.reconstruct(seq)
        if not np.allclose(reference, actual, atol=atol, rtol=rtol):
            raise RuntimeError("PyTorch/ONNX consistency check failed")
        errors.append(float(np.max(np.abs(reference - actual))))
    if not errors:
        raise ValueError("Parity requires held-out sequences")
    return {
        "passed": True,
        "max_absolute_error": max(errors),
        "sequences": len(errors),
        "atol": atol,
        "rtol": rtol,
    }
