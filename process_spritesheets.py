
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ==========================================
# CONFIGURAÇÕES
# ==========================================

INPUT_DIR = Path("GamesSprites")
OUTPUT_DIR = Path("GamesSpritesProcessed")

MAX_WIDTH = 74
MAX_HEIGHT = 100

# Quantidade mínima de pixels transparentes
# para reconhecer uma separação entre frames.
MIN_GAP = 1

# Ignorar frames vazios
SKIP_EMPTY = True

# Exigir canal alpha com transparência real
REQUIRE_TRANSPARENCY = True

# Relatório
REPORT_FILE = OUTPUT_DIR / "processing_report.json"


# ==========================================
# IDENTIFICAR REGIÕES
# ==========================================

def find_regions(projection):
    """
    Encontra regiões ocupadas separadas por
    espaços transparentes.
    """

    occupied = projection > 0

    regions = []
    start = None
    empty_count = 0

    for i, value in enumerate(occupied):

        if value:
            if start is None:
                start = i

            empty_count = 0

        elif start is not None:

            empty_count += 1

            if empty_count >= MIN_GAP:
                end = i - empty_count + 1

                regions.append((start, end))

                start = None
                empty_count = 0

    if start is not None:
        regions.append((start, len(occupied)))

    return regions


# ==========================================
# EXTRAIR FRAMES DA SPRITESHEET
# ==========================================

def extract_frames(image):

    image = image.convert("RGBA")

    pixels = np.array(image)

    alpha = pixels[:, :, 3]

    # Pixels efetivamente visíveis
    visible = alpha > 0

    # Projeções horizontal e vertical
    row_projection = visible.sum(axis=1)
    col_projection = visible.sum(axis=0)

    # Identifica linhas ocupadas
    row_regions = find_regions(row_projection)

    frames = []

    for row_start, row_end in row_regions:

        # Recorta a faixa vertical
        row_mask = visible[row_start:row_end, :]

        # Identifica colunas ocupadas nessa faixa
        col_projection = row_mask.sum(axis=0)

        col_regions = find_regions(col_projection)

        for col_start, col_end in col_regions:

            # Região candidata
            crop = image.crop((
                col_start,
                row_start,
                col_end,
                row_end
            ))

            # Remove transparência excedente
            bbox = crop.getchannel("A").getbbox()

            if bbox is None:
                continue

            crop = crop.crop(bbox)

            width, height = crop.size

            # Limite máximo
            if width > MAX_WIDTH or height > MAX_HEIGHT:
                continue

            frames.append(crop)

    return frames


# ==========================================
# PROCESSAR SPRITESHEET
# ==========================================

def process_image(image_path):

    relative_path = image_path.relative_to(INPUT_DIR)

    output_folder = OUTPUT_DIR / relative_path.parent

    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        image = Image.open(image_path)

        # Verifica se existe transparência real
        rgba = image.convert("RGBA")

        if REQUIRE_TRANSPARENCY:
            if rgba.getchannel("A").getextrema() == (255, 255):

                return {
                    "file": str(relative_path),
                    "status": "ignored_no_transparency",
                    "frames": 0
                }

        frames = extract_frames(rgba)

        if not frames:
            return {
                "file": str(relative_path),
                "status": "no_valid_frames",
                "frames": 0
            }

        saved = 0

        for index, frame in enumerate(frames, start=1):

            filename = (
                f"{image_path.stem}_{index:03d}.png"
            )

            output_path = output_folder / filename

            frame.save(output_path, "PNG")

            saved += 1

        return {
            "file": str(relative_path),
            "status": "processed",
            "frames": saved,
            "dimensions": [
                list(frame.size) for frame in frames
            ]
        }

    except Exception as error:

        return {
            "file": str(relative_path),
            "status": "error",
            "message": str(error),
            "frames": 0
        }


# ==========================================
# EXECUÇÃO
# ==========================================

def main():

    if not INPUT_DIR.exists():
        print(f"Pasta não encontrada: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(INPUT_DIR.rglob("*.png"))

    print(f"Spritesheets encontradas: {len(files)}")

    results = []

    total_frames = 0

    for file in tqdm(files, desc="Extraindo frames"):

        result = process_image(file)

        results.append(result)

        total_frames += result["frames"]

    # Relatório
    with open(REPORT_FILE, "w", encoding="utf-8") as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    processed = sum(
        r["status"] == "processed" for r in results
    )

    ignored = sum(
        r["status"] == "ignored_no_transparency"
        for r in results
    )

    errors = sum(
        r["status"] == "error" for r in results
    )

    print("\n========== RESULTADO ==========")
    print(f"Spritesheets: {len(files)}")
    print(f"Processadas: {processed}")
    print(f"Ignoradas: {ignored}")
    print(f"Erros: {errors}")
    print(f"Frames extraídos: {total_frames}")
    print(f"Saída: {OUTPUT_DIR.resolve()}")
    print("================================")


if __name__ == "__main__":
    main()
