import json
import torch

from text_encoder import TextEncoder
from config import DEVICE


def main():
    with open(
        "dataset/processed/captions.json",
        "r",
        encoding="utf-8"
    ) as file:
        captions = json.load(file)

    encoder = TextEncoder(device=DEVICE)

    # Testa as três primeiras descrições.
    test_captions = captions

    embeddings, attention_mask = encoder(test_captions)

    print("\nResultado:")
    print("Quantidade de descrições:", len(test_captions))
    print("Embeddings:", embeddings.shape)
    print("Attention mask:", attention_mask.shape)
    print("Embedding dimension:", encoder.embedding_dim)
    print("Device:", embeddings.device)
    print("Data type:", embeddings.dtype)

    assert embeddings.shape == (3, 77, 512)
    assert attention_mask.shape == (3, 77)

    print("\nText Encoder funcionando corretamente.")


if __name__ == "__main__":
    main()
