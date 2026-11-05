
import os
from contextlib import nullcontext

import torch
import torch.nn as nn

from model import PixelTransformer
from text_encoder import TextEncoder
from dataloader import get_dataloader, PixelDataset
from torch.utils.data import DataLoader

from config import (
    DEVICE,
    VOCAB_SIZE,
    SEQUENCE_LENGTH
)


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 500
BATCH_SIZE = 1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

TEXT_BATCH_SIZE = 32

CHECKPOINT_DIR = "checkpoints"

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "pixel_transformer.pt"
)

SAVE_EVERY = 50

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# TEXT EMBEDDING CACHE
# ============================================================

def precompute_text_embeddings(dataset, encoder):

    print("\nPrecomputing text embeddings...")

    unique_captions = list(
        dict.fromkeys(dataset.captions)
    )

    text_cache = {}

    encoder.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(unique_captions),
            TEXT_BATCH_SIZE
        ):

            batch_captions = unique_captions[
                start:start + TEXT_BATCH_SIZE
            ]

            embeddings, attention_mask = encoder(
                batch_captions
            )

            for i, caption in enumerate(batch_captions):

                text_cache[caption] = (
                    embeddings[i].cpu().contiguous(),
                    attention_mask[i].cpu().contiguous()
                )

            print(
                f"Encoded: "
                f"{min(start + TEXT_BATCH_SIZE, len(unique_captions))}"
                f"/{len(unique_captions)}"
            )

    print(
        f"Text embeddings cached: {len(text_cache)}"
    )

    return text_cache


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    loss
):

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "loss": loss
        },
        CHECKPOINT_PATH
    )

    print(
        f"Checkpoint saved: epoch {epoch}"
    )


# ============================================================
# TRAINING
# ============================================================

def main():

    device = torch.device(DEVICE)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível.")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = PixelDataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False
    )

    # --------------------------------------------------------
    # CLIP Text Encoder
    # --------------------------------------------------------

    text_encoder = TextEncoder(
        device=str(device)
    )

    # Congela o CLIP.

    text_encoder.eval()

    for parameter in text_encoder.parameters():
        parameter.requires_grad = False

    # Calcula os embeddings uma única vez.

    text_cache = precompute_text_embeddings(
        dataset,
        text_encoder
    )

    # --------------------------------------------------------
    # Pixel Transformer
    # --------------------------------------------------------

    model = PixelTransformer().to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Mixed Precision
    # --------------------------------------------------------

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTraining configuration:")
    print("Device:", device)
    print("Images:", len(dataset))
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Vocabulary:", VOCAB_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Trainable parameters:", trainable_parameters)
    print("Mixed precision:", use_amp)

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for inputs, targets, captions in loader:

            inputs = inputs.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            # Recupera os embeddings correspondentes
            # às descrições deste batch.

            embeddings = torch.stack([
                text_cache[caption][0]
                for caption in captions
            ]).to(
                device,
                non_blocking=True
            )

            attention_mask = torch.stack([
                text_cache[caption][1]
                for caption in captions
            ]).to(
                device,
                non_blocking=True
            )

            # Zera gradientes.

            optimizer.zero_grad(
                set_to_none=True
            )

            # Forward pass.

            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                )
                if use_amp
                else nullcontext()
            )

            with autocast_context:

                logits = model(
                    inputs,
                    embeddings,
                    attention_mask
                )

                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    targets.reshape(-1)
                )

            # Backpropagation.

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)

            scaler.update()

            total_loss += loss.item()

        # ----------------------------------------------------
        # Epoch Statistics
        # ----------------------------------------------------

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {average_loss:.5f}"
        )

        # ----------------------------------------------------
        # Save Checkpoint
        # ----------------------------------------------------

        if (epoch + 1) % SAVE_EVERY == 0:

            save_checkpoint(
                model,
                optimizer,
                scaler,
                epoch + 1,
                average_loss
            )

    # --------------------------------------------------------
    # Final Checkpoint
    # --------------------------------------------------------

    save_checkpoint(
        model,
        optimizer,
        scaler,
        EPOCHS,
        average_loss
    )

    print("\nTraining concluído.")


if __name__ == "__main__":
    main()
