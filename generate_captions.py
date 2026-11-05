
import os
import json
import base64
import mimetypes
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests


# ==========================================
# CONFIGURAÇÕES
# ==========================================

INPUT_DIR = Path("dataset/originals")
OUTPUT_FILE = Path("captions.json")

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# Substitua pelo identificador do modelo carregado no LM Studio.
MODEL = "qwen/qwen3-vl-30b"

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp"
}

MAX_TOKENS = 160
UPSCALE_FACTOR = 3  # 300%


# ==========================================
# PROMPT
# ==========================================

SYSTEM_PROMPT = """
You are an expert pixel art analyst and dataset annotator.

Your task is to describe pixel art images accurately for a
machine learning image generation dataset.

Analyze the image and produce a detailed but concise visual
description.

Include relevant information about:

1. Main subject: character, creature, object, environment.
2. Physical appearance and distinctive characteristics.
3. Clothing, armor, accessories and equipment.
4. Pose, posture and orientation.
5. Dominant colors and color palette.
6. Secondary objects and visual elements.
7. Background and environment.
8. Pixel art characteristics, such as pixel clusters,
   outlines, shading and visual style.

IMPORTANT RULES:

- Describe only what is actually visible.
- Never invent names, identities, stories or characteristics.
- Do not make assumptions about elements that are unclear.
- Prioritize visual characteristics over subjective opinions.
- Preserve important details that distinguish the subject.
- Avoid generic descriptions.
- Do not mention image resolution or file information.
- Write a natural, coherent description in English.
- Return ONLY the description, without titles or explanations.
"""


SYSTEM_PROMPT = """
You are a pixel art image captioning assistant.

Describe the image using a short, objective visual description.

RULES:
- Use between 20 and 90 words.
- Describe only visible elements.
- Start with the main subject.
- Include distinctive physical features, colors, clothing,
  weapons, accessories or equipment when visible.
- Use simple, direct English.
- Separate visual characteristics using commas.
- Do not describe artistic techniques or image quality.
- Do not describe the background unless it is essential.
- Do not add introductions, explanations or titles.
- Never invent details.

Return ONLY the caption.
"""

SYSTEM_PROMPT = """
You are an expert pixel art character captioning assistant.

Analyze the image and generate a short, objective description
of the main character.

Follow this priority:

1. CHARACTER TYPE:
   Identify what the character is, such as warrior, knight,
   orc, skeleton, demon, animal, monster or human.

2. BODY COLOR AND APPEARANCE:
   Describe the main colors, body shape, size and physical appearance.

3. DISTINCTIVE FEATURES:
   Mention recognizable characteristics such as horns, wings,
   glowing eyes, scars, tails, spikes, unusual anatomy or facial features.

4. EQUIPMENT AND ACCESSORIES:
   Describe visible weapons, armor, clothing, shields,
   helmets and other accessories.

5. POSE:
   Mention the character's posture, orientation or action
   only when it is visually relevant.

STRICT RULES:
- Use 20 to 40 words.
- Write in simple, objective English.
- Use comma-separated visual characteristics.
- Prioritize the most visually distinctive details.
- Do not invent or assume missing information.
- Do not include character names, stories or lore.
- Do not describe image quality or artistic techniques.
- Do not describe the background.
- Omit categories that are not applicable.
- Do not use titles, labels or explanations.

Return ONLY the description in a single line.
"""


# ==========================================
# FUNÇÕES
# ==========================================

def load_captions():
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return {}


def save_captions(captions):
    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(
            captions,
            file,
            indent=4,
            ensure_ascii=False
        )

    temp_file.replace(OUTPUT_FILE)


def encode_image(image_path):
    with Image.open(image_path) as img:

        # Preserva o modo de cor original
        img = img.convert("RGB")

        new_width = img.width * UPSCALE_FACTOR
        new_height = img.height * UPSCALE_FACTOR

        # Ampliação sem interpolação
        img = img.resize(
            (new_width, new_height),
            Image.Resampling.NEAREST
        )

        # Codifica diretamente em memória
        buffer = BytesIO()
        img.save(buffer, format="PNG")

        return base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")


def generate_caption(image_path):
    image_base64 = encode_image(image_path)

    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None:
        mime_type = "image/png"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this pixel art image and describe its visual content."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
        "stream": False
    }

    response = requests.post(
        LM_STUDIO_URL,
        json=payload,
        timeout=180
    )
    
    if not response.ok:
        print("\n========== LM STUDIO ERROR ==========")
        print("Status:", response.status_code)
        print("Response:", response.text)
        print("=====================================\n")

        response.raise_for_status()

    response.raise_for_status()

    result = response.json()

    caption = result["choices"][0]["message"]["content"]

    if isinstance(caption, list):
        caption = " ".join(
            item.get("text", "")
            for item in caption
            if isinstance(item, dict)
        )

    return caption.strip()


# ==========================================
# PROCESSAMENTO
# ==========================================

def main():
    captions = load_captions()

    images = sorted(
        path
        for path in INPUT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    total = len(images)

    print(f"Images found: {total}")
    print(f"Existing captions: {len(captions)}")
    print("-" * 50)

    processed = 0
    errors = 0

    for index, image_path in enumerate(images, start=1):

        # Caminho relativo para evitar colisões entre
        # imagens com o mesmo nome em subpastas diferentes.
        image_name = image_path.relative_to(INPUT_DIR).as_posix()

        if image_name in captions:
            continue

        print(f"[{index}/{total}] Processing: {image_name}")

        try:
            caption = generate_caption(image_path)

            if not caption:
                raise ValueError("Empty caption returned by model")

            captions[image_name] = caption

            # Salva imediatamente após cada imagem.
            save_captions(captions)

            processed += 1

            print(f"Caption: {caption[:150]}...")
            print("Saved successfully!\n")

        except Exception as error:
            errors += 1
            print(f"ERROR: {error}\n")

    print("-" * 50)
    print("PROCESSING FINISHED")
    print(f"Total images: {total}")
    print(f"New captions: {processed}")
    print(f"Errors: {errors}")
    print(f"Total captions in JSON: {len(captions)}")


if __name__ == "__main__":
    main()
