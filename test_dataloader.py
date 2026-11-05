
import torch

from dataloader import get_dataloader
from config import SEQUENCE_LENGTH


def main():

    dataloader = get_dataloader(
        shuffle=False,
        batch_size=1
    )

    print("\nTestando DataLoader...\n")

    for batch_idx, (input_tokens, target_tokens, captions) in enumerate(dataloader):

        print(f"Batch: {batch_idx}")

        print("Input:", input_tokens.shape)
        print("Target:", target_tokens.shape)
        print("Caption:", captions[0])

        assert input_tokens.shape == (
            1,
            SEQUENCE_LENGTH - 1
        )

        assert target_tokens.shape == (
            1,
            SEQUENCE_LENGTH - 1
        )

        assert len(captions) == 1

        assert isinstance(captions[0], str)

        # Verifica o deslocamento dos tokens.

        assert torch.equal(
            input_tokens[:, 1:],
            target_tokens[:, :-1]
        )

        print("Validação do batch concluída.\n")

        # Testa apenas o primeiro batch.

        if batch_idx == 0:
            break

    print("DataLoader funcionando corretamente.")


if __name__ == "__main__":
    main()
