import os
import json
import glob

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    PALETTE_SIZE,
    TRANSPARENT_TOKEN,
    BOS_TOKEN,
    VOCAB_SIZE
)

INPUT_DIR = "dataset/originals"
OUTPUT_DIR = "dataset/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_images():
    paths = glob.glob(os.path.join(INPUT_DIR, "*.png"))

    if not paths:
        raise RuntimeError("Nenhuma imagem PNG encontrada.")

    images = []

    for path in paths:
        image = Image.open(path).convert("RGBA")

        image.thumbnail(
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            Image.Resampling.NEAREST
        )

        canvas = Image.new(
            "RGBA",
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            (0, 0, 0, 0)
        )

        x = (IMAGE_WIDTH - image.width) // 2
        y = (IMAGE_HEIGHT - image.height) // 2

        canvas.alpha_composite(image, (x, y))

        images.append((path, np.array(canvas)))

    return images


def create_palette(images):
    samples = []

    for _, image in images:
        pixels = image.reshape(-1, 4)
        opaque = pixels[pixels[:, 3] >= 128, :3]

        if len(opaque):
            samples.append(opaque)

    if not samples:
        raise RuntimeError("Não existem pixels opacos.")

    samples = np.concatenate(samples)

    if len(samples) > 200000:
        indices = np.random.choice(
            len(samples),
            200000,
            replace=False
        )
        samples = samples[indices]

    kmeans = MiniBatchKMeans(
        n_clusters=PALETTE_SIZE,
        random_state=42,
        batch_size=4096,
        n_init=3
    )

    kmeans.fit(samples)

    return np.clip(
        np.rint(kmeans.cluster_centers_),
        0,
        255
    ).astype(np.uint8)


def encode_image(image, palette):
    pixels = image[:, :, :3].astype(np.float32)
    alpha = image[:, :, 3]

    palette_float = palette.astype(np.float32)

    distances = (
        (pixels[:, :, None, :] - palette_float[None, None, :, :]) ** 2
    ).sum(axis=3)

    color_indices = distances.argmin(axis=2) + 1

    tokens = color_indices.astype(np.uint8)

    tokens[alpha < 128] = TRANSPARENT_TOKEN

    return tokens.flatten()


def main():
    images = load_images()

    print(f"Imagens encontradas: {len(images)}")

    palette = create_palette(images)

    sequences = []

    for path, image in tqdm(images):
        tokens = encode_image(image, palette)

        sequence = np.concatenate([
            np.array([BOS_TOKEN], dtype=np.uint8),
            tokens
        ])

        sequences.append(sequence)

    sequences = np.stack(sequences)

    assert sequences.max() < VOCAB_SIZE
    assert sequences.shape[1] == IMAGE_WIDTH * IMAGE_HEIGHT + 1

    np.save(
        os.path.join(OUTPUT_DIR, "tokens.npy"),
        sequences
    )

    palette_data = {
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "palette_size": PALETTE_SIZE,
        "bos_token": BOS_TOKEN,
        "colors": palette.tolist()
    }

    with open(
        os.path.join(OUTPUT_DIR, "palette.json"),
        "w"
    ) as file:
        json.dump(palette_data, file, indent=2)

    print("\nDataset preparado.")
    print("Shape:", sequences.shape)
    print("Sequência por imagem:", sequences.shape[1])
    print("Paleta:", PALETTE_SIZE)
    print("Vocabulário:", VOCAB_SIZE)


if __name__ == "__main__":
    main()