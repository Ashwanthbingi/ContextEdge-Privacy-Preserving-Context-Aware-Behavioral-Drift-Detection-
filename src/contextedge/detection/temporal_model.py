from __future__ import annotations

import io
import time

import torch
from torch import nn


class GRUAutoencoder(nn.Module):
    """Sequence bottleneck followed by a shared recurrent decoder."""

    def __init__(self, n_features: int, hidden: int = 16):
        super().__init__()
        self.encoder = nn.GRU(n_features, hidden, batch_first=True)
        self.decoder = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, hidden = self.encoder(x)
        repeated = hidden[-1].unsqueeze(1).expand(-1, x.shape[1], -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def train_temporal(train_sequences, *, hidden=16, epochs=18, seed=42):
    if len(train_sequences) < 20:
        raise ValueError("At least 20 contiguous normal training sequences are required")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = GRUAutoencoder(train_sequences.shape[-1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    data = torch.as_tensor(train_sequences, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        total = 0.0
        order = torch.randperm(len(data), generator=generator)
        for batch in order.split(32):
            x = data[batch]
            optimizer.zero_grad()
            loss = torch.mean((model(x) - x) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        losses.append(total / len(data))
    model.eval()
    return model, {"epochs": epochs, "losses": losses, "training_seconds": time.perf_counter() - start}


def export_onnx(model, sequence_length: int, n_features: int) -> bytes:
    buffer = io.BytesIO()
    example = torch.zeros(1, sequence_length, n_features)
    torch.onnx.export(
        model,
        example,
        buffer,
        input_names=["features"],
        output_names=["reconstruction"],
        opset_version=17,
        dynamo=False,
    )
    return buffer.getvalue()
