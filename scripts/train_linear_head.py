"""Train the 7-class linear head on candidate_tiles.

Only the linear head is trained; the ResNet18 backbone is frozen and uses
ImageNet-pretrained weights. The pretrained backbone is downloaded ONCE on
the first run (internet required at training time). The resulting full
state_dict is saved to MODEL_WEIGHTS_PATH, and runtime inference then uses
only that local file.

Usage:
    python -m scripts.train_linear_head
"""

from __future__ import annotations

import glob

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from galaxeye.config import CANDIDATE_TILES_DIR, MODEL_WEIGHTS_PATH
from galaxeye.model import LandClass
from galaxeye.preprocess import to_input_array

CLASSES = list(LandClass)
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
SEED = 42


def load_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for index, cls in enumerate(CLASSES):
        folder = CANDIDATE_TILES_DIR / cls.value
        for path in sorted(glob.glob(str(folder / "*.png"))):
            array = to_input_array(Image.open(path))[0]  # (3, 64, 64)
            xs.append(array)
            ys.append(index)
    x = torch.from_numpy(np.stack(xs))
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def build_model(num_classes: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main() -> None:
    torch.manual_seed(SEED)
    x, y = load_dataset()
    print(f"Loaded {x.shape[0]} tiles across {len(CLASSES)} classes.")

    model = build_model(len(CLASSES))
    head = model.fc
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    n = x.shape[0]
    updates = 0
    for epoch in range(1, EPOCHS + 1):
        permutation = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            indices = permutation[i : i + BATCH_SIZE]
            batch_x, batch_y = x[indices], y[indices]
            optimizer.zero_grad()
            out = model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            optimizer.step()
            updates += 1
            epoch_loss += loss.item() * batch_x.shape[0]
        avg_loss = epoch_loss / n
        acc = (model(x).argmax(1) == y).float().mean().item()
        print(f"epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  train_acc={acc:.4f}")

    print(f"Total optimizer updates: {updates}")
    MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    print(f"Saved model artifact: {MODEL_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
