
import os
import json

import numpy as np
import torch
from PIL import Image

from model import PixelTransformer
from text_encoder import TextEncoder

from config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    PALETTE_SIZE,
    BOS_TOKEN,
    VOCAB_SIZE,
    DEVICE
)


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = "checkpoints/pixel_transformer.pt"

PALETTE_PATH = "dataset/processed/palette.json"

OUTPUT_DIR = "generated"

TEMPERATURE = 0.8
TOP_K = 8

NUM_IMAGES = 4

SCALE = 6


# ============================================================
# IMAGE GENERATION
# ============================================================

@torch.inference_mode()
def generate(
    model,
    text_embeddings,
    text_attention_mask
):

    model.eval()

    device = next(model.parameters()).device

    tokens = torch.tensor(
        [[BOS_TOKEN]],
        dtype=torch.long,
        device=device
    )

    use_amp = device.type == "cuda"

    for step in range(IMAGE_WIDTH * IMAGE_HEIGHT):

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            logits = model(
                tokens,
                text_embeddings,
                text_attention_mask
            )

        # Obtém os logits do próximo token.

        next_logits = logits[:, -1, :].float()

        next_logits = next_logits / TEMPERATURE

        # Impede a geração do token BOS.

        next_logits[:, BOS_TOKEN] = float("-inf")

        # Top-K sampling.

        if TOP_K is not None and TOP_K > 0:

            k = min(TOP_K, VOCAB_SIZE)

            values, _ = torch.topk(
                next_logits,
                k
            )

            threshold = values[:, -1].unsqueeze(-1)

            next_logits[next_logits < threshold] = float("-inf")

        # Converte logits em probabilidades.

        probabilities = torch.softmax(
            next_logits,
            dim=-1
        )

        # Seleciona o próximo token.

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1
        )

        if (step + 1) % 1000 == 0:

            print(
                f"Generated: {step + 1}/"
                f"{IMAGE_WIDTH * IMAGE_HEIGHT} pixels"
            )

    return tokens[0, 1:].cpu().numpy()


# ============================================================
# IMAGE DECODING
# ============================================================

def decode(tokens, palette):

    lookup = np.zeros(
        (VOCAB_SIZE, 4),
        dtype=np.uint8
    )

    # Tokens 1 até PALETTE_SIZE representam cores.

    lookup[1:PALETTE_SIZE + 1, :3] = palette

    lookup[1:PALETTE_SIZE + 1, 3] = 255

    pixels = lookup[tokens]

    image = Image.fromarray(
        pixels.reshape(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            4
        )
    )

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    device = torch.device(DEVICE)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível.")

    # --------------------------------------------------------
    # Load palette
    # --------------------------------------------------------

    with open(
        PALETTE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        palette_data = json.load(file)

    palette = np.array(
        palette_data["colors"],
        dtype=np.uint8
    )

    if palette.shape != (PALETTE_SIZE, 3):
        raise ValueError(
            f"Paleta inválida: {palette.shape}"
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    model = PixelTransformer().to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    print("\nModelo carregado.")

    print("Epoch:", checkpoint["epoch"])

    print("Loss:", checkpoint["loss"])

    # --------------------------------------------------------
    # Load CLIP
    # --------------------------------------------------------

    encoder = TextEncoder(
        device=str(device)
    )

    encoder.eval()

    # --------------------------------------------------------
    # Get description
    # --------------------------------------------------------

    print("\nDigite a descrição da imagem que deseja gerar:")

    prompt = input("> ").strip()

    if not prompt:
        raise ValueError("A descrição não pode estar vazia.")

    # --------------------------------------------------------
    # Encode description
    # --------------------------------------------------------

    print("\nEncoding prompt...")

    with torch.inference_mode():

        text_embeddings, text_attention_mask = encoder(
            prompt
        )

    print("Text embeddings:", text_embeddings.shape)

    # --------------------------------------------------------
    # Generate images
    # --------------------------------------------------------

    for index in range(NUM_IMAGES):

        print(
            f"\nGenerating image {index + 1}/{NUM_IMAGES}..."
        )

        tokens = generate(
            model,
            text_embeddings,
            text_attention_mask
        )

        image = decode(
            tokens,
            palette
        )

        # Nearest Neighbor upscale.

        image = image.resize(
            (
                IMAGE_WIDTH * SCALE,
                IMAGE_HEIGHT * SCALE
            ),
            Image.Resampling.NEAREST
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"generated_{index + 1:02d}.png"
        )

        image.save(output_path)

        print("Saved:", output_path)

    print("\nGeneration completed.")


if __name__ == "__main__":
    main()
