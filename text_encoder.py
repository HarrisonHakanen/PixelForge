
import torch
import torch.nn as nn

from transformers import CLIPTokenizer, CLIPTextModel


MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_LENGTH = 77


class TextEncoder(nn.Module):

    def __init__(self, device="cuda", local_files_only=False):
        super().__init__()

        self.device = torch.device(device)

        print("Carregando tokenizer CLIP...")

        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=local_files_only
        )

        print("Carregando CLIP Text Encoder...")

        self.encoder = CLIPTextModel.from_pretrained(
            MODEL_NAME,
            local_files_only=local_files_only
        )

        self.encoder.to(self.device)

        # Congela todos os parâmetros do CLIP.
        self.encoder.eval()

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.embedding_dim = self.encoder.config.hidden_size

        print("Text Encoder carregado.")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Max sequence length: {MAX_LENGTH}")

    @torch.no_grad()
    def forward(self, captions):
        """
        Recebe uma lista de descrições e retorna:

        embeddings:
            Tensor [batch_size, 77, 512]

        attention_mask:
            Tensor [batch_size, 77]
        """

        if isinstance(captions, str):
            captions = [captions]

        if not captions:
            raise ValueError("A lista de descrições está vazia.")

        inputs = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        embeddings = outputs.last_hidden_state

        return embeddings, attention_mask