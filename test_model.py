
import torch

from config import (
    DEVICE,
    SEQUENCE_LENGTH,
    VOCAB_SIZE,
    TEXT_EMBEDDING_DIM
)

from text_encoder import TextEncoder
from model import PixelTransformer


def main():

    captions = [
        "Small green orc, muscular body, holding a wooden axe",
        "Red dragon with large wings and sharp claws",
        "Small blue slime monster with round body"
    ]

    # CLIP Text Encoder

    encoder = TextEncoder(device=DEVICE)

    embeddings, attention_mask = encoder(captions)

    # Pixel Transformer

    model = PixelTransformer().to(DEVICE)

    # Simula uma sequência de tokens de pixels.

    tokens = torch.randint(
        0,
        VOCAB_SIZE,
        (3, SEQUENCE_LENGTH),
        device=DEVICE
    )

    # Forward pass

    with torch.no_grad():

        logits = model(
            tokens,
            embeddings,
            attention_mask
        )

    print("\nResultado:")

    print("Tokens:", tokens.shape)

    print("Text embeddings:", embeddings.shape)

    print("Attention mask:", attention_mask.shape)

    print("Logits:", logits.shape)

    print("Device:", logits.device)

    assert logits.shape == (
        3,
        SEQUENCE_LENGTH,
        VOCAB_SIZE
    )

    print("\nPixelTransformer funcionando corretamente.")


if __name__ == "__main__":
    main()
