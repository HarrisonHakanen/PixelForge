
import os
import glob
import json

import numpy as np
from PIL import Image

from config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    PALETTE_SIZE,
    VOCAB_SIZE
)

DATASET_PATH = "dataset/processed/tokens.npy"
PALETTE_PATH = "dataset/processed/palette.json"
INPUT_DIR = "dataset/originals"
OUTPUT_DIR = "reconstruction_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def decode(tokens, palette):
    lookup = np.zeros((VOCAB_SIZE, 4), dtype=np.uint8)

    lookup[1:PALETTE_SIZE + 1, :3] = palette
    lookup[1:PALETTE_SIZE + 1, 3] = 255

    pixels = lookup[tokens]

    return Image.fromarray(
        pixels.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 4),
        mode="RGBA"
    )


def main():
    sequences = np.load(DATASET_PATH)

    with open(PALETTE_PATH) as file:
        palette_data = json.load(file)

    palette = np.array(
        palette_data["colors"],
        dtype=np.uint8
    )

    paths = glob.glob(os.path.join(INPUT_DIR, "*.png"))

    print("Imagens no dataset:", len(sequences))
    print("Imagens originais:", len(paths))

    if len(sequences) != len(paths):
        raise RuntimeError("Quantidade de imagens incompatível.")

    for index, path in enumerate(paths):

        # Remove o BOS e recupera os tokens dos pixels.
        tokens = sequences[index, 1:].astype(np.int64)

        reconstructed = decode(tokens, palette)

        # Amplia sem interpolação para facilitar a inspeção.
        reconstructed = reconstructed.resize(
            (IMAGE_WIDTH * 6, IMAGE_HEIGHT * 6),
            Image.Resampling.NEAREST
        )

        filename = f"reconstruction_{index + 1:02d}.png"

        reconstructed.save(
            os.path.join(OUTPUT_DIR, filename)
        )

        print(f"Reconstruída: {filename}")

    print("\nTeste concluído.")
    print("Resultados em:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
