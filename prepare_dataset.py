
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
CAPTIONS_PATH = "dataset/captions.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_captions():
    if not os.path.exists(CAPTIONS_PATH):
        raise FileNotFoundError(
            f"Arquivo de descrições não encontrado: {CAPTIONS_PATH}"
        )

    with open(CAPTIONS_PATH, "r", encoding="utf-8") as file:
        captions = json.load(file)

    if not isinstance(captions, dict):
        raise ValueError(
            "captions.json precisa conter um objeto JSON "
            "com nome do arquivo e descrição."
        )

    for filename, description in captions.items():
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"Descrição inválida ou vazia para: {filename}"
            )

    return captions


def load_images(captions):
    paths = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.png"))
    )

    if not paths:
        raise RuntimeError("Nenhuma imagem PNG encontrada.")

    image_names = {
        os.path.basename(path)
        for path in paths
    }

    caption_names = set(captions.keys())

    missing_captions = image_names - caption_names
    missing_images = caption_names - image_names

    if missing_captions or missing_images:
        errors = []

        if missing_captions:
            errors.append(
                "Imagens sem descrição:\n"
                + "\n".join(sorted(missing_captions))
            )

        if missing_images:
            errors.append(
                "Descrições sem imagem:\n"
                + "\n".join(sorted(missing_images))
            )

        raise ValueError("\n\n".join(errors))

    images = []

    for path in paths:
        filename = os.path.basename(path)

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

        images.append({
            "filename": filename,
            "image": np.array(canvas),
            "caption": captions[filename].strip()
        })

    return images


def create_palette(images):
    samples = []

    for item in images:
        image = item["image"]

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
        (
            pixels[:, :, None, :]
            - palette_float[None, None, :, :]
        ) ** 2
    ).sum(axis=3)

    color_indices = distances.argmin(axis=2) + 1

    tokens = color_indices.astype(np.uint8)

    tokens[alpha < 128] = TRANSPARENT_TOKEN

    return tokens.flatten()


def save_dataset(sequences, captions, filenames, palette):
    np.save(
        os.path.join(OUTPUT_DIR, "tokens.npy"),
        sequences
    )

    with open(
        os.path.join(OUTPUT_DIR, "captions.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            captions,
            file,
            indent=2,
            ensure_ascii=False
        )

    metadata = {
        "num_images": len(filenames),
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "sequence_length": IMAGE_WIDTH * IMAGE_HEIGHT + 1,
        "palette_size": PALETTE_SIZE,
        "vocab_size": VOCAB_SIZE,
        "bos_token": BOS_TOKEN,
        "transparent_token": TRANSPARENT_TOKEN,
        "filenames": filenames
    }

    with open(
        os.path.join(OUTPUT_DIR, "metadata.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
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
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            palette_data,
            file,
            indent=2
        )


def main():
    captions_dict = load_captions()

    images = load_images(captions_dict)

    print(f"Imagens encontradas: {len(images)}")
    print("Descrições validadas.")

    palette = create_palette(images)

    sequences = []
    captions = []
    filenames = []

    for item in tqdm(images, desc="Codificando imagens"):
        tokens = encode_image(item["image"], palette)

        sequence = np.concatenate([
            np.array([BOS_TOKEN], dtype=np.uint8),
            tokens
        ])

        sequences.append(sequence)
        captions.append(item["caption"])
        filenames.append(item["filename"])

    sequences = np.stack(sequences)

    assert sequences.max() < VOCAB_SIZE

    assert sequences.shape[1] == (
        IMAGE_WIDTH * IMAGE_HEIGHT + 1
    )

    save_dataset(
        sequences,
        captions,
        filenames,
        palette
    )

    print("\nDataset preparado com sucesso.")
    print(f"Imagens: {len(images)}")
    print(f"Shape dos tokens: {sequences.shape}")
    print(f"Descrições: {len(captions)}")
    print(f"Sequência por imagem: {sequences.shape[1]}")
    print(f"Paleta: {PALETTE_SIZE}")
    print(f"Vocabulário: {VOCAB_SIZE}")

    print("\nArquivos gerados:")
    print("- dataset/processed/tokens.npy")
    print("- dataset/processed/captions.json")
    print("- dataset/processed/metadata.json")
    print("- dataset/processed/palette.json")


if __name__ == "__main__":
    main()
