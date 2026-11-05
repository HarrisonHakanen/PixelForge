
import os
import json

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from config import (
    SEQUENCE_LENGTH,
    VOCAB_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE
)


PROCESSED_DIR = "dataset/processed"

TOKENS_PATH = os.path.join(
    PROCESSED_DIR,
    "tokens.npy"
)

CAPTIONS_PATH = os.path.join(
    PROCESSED_DIR,
    "captions.json"
)

METADATA_PATH = os.path.join(
    PROCESSED_DIR,
    "metadata.json"
)


class PixelDataset(Dataset):

    def __init__(self):

        print("Carregando dataset...")

        # Carrega as sequências de tokens.

        self.tokens = np.load(
            TOKENS_PATH
        )

        # Carrega as descrições.

        with open(
            CAPTIONS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            self.captions = json.load(file)

        # Carrega os metadados.

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)

        self.filenames = self.metadata["filenames"]

        self._validate()

        print("Dataset carregado.")
        print(f"Quantidade de imagens: {len(self.tokens)}")
        print(f"Sequência por imagem: {SEQUENCE_LENGTH}")

    def _validate(self):

        if len(self.tokens) == 0:
            raise ValueError("Dataset de tokens vazio.")

        if self.tokens.ndim != 2:
            raise ValueError(
                "tokens.npy deve possuir duas dimensões."
            )

        if self.tokens.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequência inválida: {self.tokens.shape[1]}. "
                f"Esperado: {SEQUENCE_LENGTH}."
            )

        if len(self.captions) != len(self.tokens):
            raise ValueError(
                "Quantidade de descrições diferente "
                "da quantidade de imagens."
            )

        if len(self.filenames) != len(self.tokens):
            raise ValueError(
                "Quantidade de filenames diferente "
                "da quantidade de imagens."
            )

        if not all(
            isinstance(caption, str) and caption.strip()
            for caption in self.captions
        ):
            raise ValueError(
                "Existem descrições vazias ou inválidas."
            )

        if not np.issubdtype(
            self.tokens.dtype,
            np.integer
        ):
            raise ValueError(
                "Os tokens devem possuir valores inteiros."
            )

        if self.tokens.min() < 0 or self.tokens.max() >= VOCAB_SIZE:
            raise ValueError(
                "Existem tokens fora do vocabulário configurado."
            )

        print("Validação dos dados concluída.")

    def __len__(self):

        return len(self.tokens)

    def __getitem__(self, index):

        sequence = torch.as_tensor(
            self.tokens[index],
            dtype=torch.long
        )

        # Entrada: todos os tokens, exceto o último.

        input_tokens = sequence[:-1]

        # Target: todos os tokens, exceto o primeiro.

        target_tokens = sequence[1:]

        caption = self.captions[index]

        return (
            input_tokens,
            target_tokens,
            caption
        )


def get_dataloader(
    shuffle=True,
    batch_size=None
):

    dataset = PixelDataset()

    if batch_size is None:
        batch_size = BATCH_SIZE

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=str(DEVICE).startswith("cuda"),
        drop_last=False
    )

    return dataloader
