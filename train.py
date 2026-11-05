import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from model import PixelTransformer

from config import (
    DEVICE,
    VOCAB_SIZE,
    SEQUENCE_LENGTH
)


EPOCHS = 500
BATCH_SIZE = 1
LEARNING_RATE = 3e-4

DATASET_PATH = "dataset/processed/tokens.npy"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class PixelDataset(Dataset):

    def __init__(self, path):
        self.data = np.load(path)

        if self.data.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequência incorreta: {self.data.shape[1]}. "
                f"Esperado: {SEQUENCE_LENGTH}."
            )

        if self.data.max() >= VOCAB_SIZE:
            raise ValueError("Dataset contém tokens fora do vocabulário.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = torch.tensor(
            self.data[index].astype(np.int64),
            dtype=torch.long
        )

        return sequence[:-1], sequence[1:]


def main():

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível.")

    dataset = PixelDataset(DATASET_PATH)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    model = PixelTransformer().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE == "cuda")
    )

    print("Device:", DEVICE)
    print("Images:", len(dataset))
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Vocabulary:", VOCAB_SIZE)
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for inputs, targets in loader:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE,
                dtype=torch.float16,
                enabled=(DEVICE == "cuda")
            ):
                logits = model(inputs)

                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    targets.reshape(-1)
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {average_loss:.5f}"
        )

        if (epoch + 1) % 50 == 0:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch + 1,
                    "loss": average_loss
                },
                os.path.join(
                    CHECKPOINT_DIR,
                    "pixel_transformer.pt"
                )
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": EPOCHS,
            "loss": average_loss
        },
        os.path.join(
            CHECKPOINT_DIR,
            "pixel_transformer.pt"
        )
    )

    print("Training concluído.")


if __name__ == "__main__":
    main()