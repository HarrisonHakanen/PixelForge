import os
import json

import numpy as np
import torch
from PIL import Image

from model import PixelTransformer

from config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    PALETTE_SIZE,
    BOS_TOKEN,
    VOCAB_SIZE,
    DEVICE
)


CHECKPOINT = "checkpoints/pixel_transformer.pt"
PALETTE_PATH = "dataset/processed/palette.json"

OUTPUT_DIR = "generated"

TEMPERATURE = 0.8
TOP_K = 8
NUM_IMAGES = 8


@torch.no_grad()
def generate(model):

    model.eval()

    tokens = torch.tensor(
        [[BOS_TOKEN]],
        dtype=torch.long,
        device=DEVICE
    )

    for _ in range(IMAGE_WIDTH * IMAGE_HEIGHT):

        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.float16,
            enabled=(DEVICE == "cuda")
        ):
            logits = model(tokens)

        next_logits = logits[:, -1, :] / TEMPERATURE

        next_logits[:, BOS_TOKEN] = float("-inf")

        if TOP_K:
            values, _ = torch.topk(next_logits, TOP_K)
            threshold = values[:, -1].unsqueeze(-1)
            next_logits[next_logits < threshold] = float("-inf")
        '''
        probabilities = torch.softmax(next_logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )
        '''
        next_token = torch.argmax(
            next_logits,
            dim=-1,
            keepdim=True
        )

        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens[0, 1:].cpu().numpy()


def decode(tokens, palette):

    lookup = np.zeros(
        (VOCAB_SIZE, 4),
        dtype=np.uint8
    )

    lookup[1:PALETTE_SIZE + 1, :3] = palette
    lookup[1:PALETTE_SIZE + 1, 3] = 255

    pixels = lookup[tokens]

    return Image.fromarray(
        pixels.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 4),
        mode="RGBA"
    )


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PALETTE_PATH) as file:
        palette_data = json.load(file)

    palette = np.array(
        palette_data["colors"],
        dtype=np.uint8
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model = PixelTransformer().to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])

    print("Modelo carregado.")
    print("Epoch:", checkpoint["epoch"])
    print("Loss:", checkpoint["loss"])

    for index in range(NUM_IMAGES):

        tokens = generate(model)

        image = decode(tokens, palette)

        image = image.resize(
            (IMAGE_WIDTH * 6, IMAGE_HEIGHT * 6),
            Image.Resampling.NEAREST
        )

        image.save(
            os.path.join(
                OUTPUT_DIR,
                f"monster_{index + 1:02d}.png"
            )
        )

        print(f"Imagem {index + 1} gerada.")


if __name__ == "__main__":
    main()