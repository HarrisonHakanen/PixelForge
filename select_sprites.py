
import csv
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# ==========================================
# CONFIGURAÇÕES
# ==========================================

INPUT_DIR = Path("sprites_processed")
OUTPUT_DIR = Path("dataset_selected")

IMAGES_PER_FOLDER = 5

# Filtros de qualidade
MIN_OPAQUE_PIXELS = 80
MIN_BBOX_AREA = 100
MIN_DENSITY = 0.10

IMAGE_EXTENSIONS = {".png"}


# ==========================================
# ANALISAR SPRITE
# ==========================================

def analyze_sprite(image_path):

    try:
        image = Image.open(image_path).convert("RGBA")

        alpha = image.getchannel("A")

        bbox = alpha.getbbox()

        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        # Quantidade de pixels visíveis
        opaque_pixels = sum(
            1 for pixel in alpha.getdata()
            if pixel > 0
        )

        # Área ocupada pelo conteúdo
        bbox_area = width * height

        # Densidade de pixels
        density = opaque_pixels / bbox_area

        return {
            "path": image_path,
            "width": width,
            "height": height,
            "opaque_pixels": opaque_pixels,
            "bbox_area": bbox_area,
            "density": density
        }

    except Exception as error:

        print(f"[ERRO] {image_path}: {error}")

        return None


# ==========================================
# VALIDAR QUALIDADE
# ==========================================

def is_valid_sprite(sprite):

    if sprite["opaque_pixels"] < MIN_OPAQUE_PIXELS:
        return False

    if sprite["bbox_area"] < MIN_BBOX_AREA:
        return False

    if sprite["density"] < MIN_DENSITY:
        return False

    return True


# ==========================================
# PROCESSAR PASTAS
# ==========================================

def main():

    if not INPUT_DIR.exists():
        print(f"Pasta não encontrada: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Localiza todas as imagens
    image_files = [
        p for p in INPUT_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Agrupa por pasta
    folders = {}

    for image_path in image_files:

        relative_folder = image_path.parent.relative_to(INPUT_DIR)

        folders.setdefault(relative_folder, []).append(image_path)

    print(f"\nPastas encontradas: {len(folders)}")
    print(f"Sprites encontradas: {len(image_files)}")

    selected_total = 0
    rejected_total = 0

    report = []

    # Processa cada personagem
    for folder, images in tqdm(
        folders.items(),
        desc="Analisando personagens"
    ):

        candidates = []

        for image_path in images:

            sprite = analyze_sprite(image_path)

            if sprite is None:
                rejected_total += 1
                continue

            sprite["valid"] = is_valid_sprite(sprite)

            candidates.append(sprite)

        # Ordena pela quantidade de pixels visíveis
        candidates.sort(
            key=lambda s: (
                s["opaque_pixels"],
                s["bbox_area"],
                s["density"]
            ),
            reverse=True
        )

        valid_candidates = [
            s for s in candidates if s["valid"]
        ]

        selected = valid_candidates[:IMAGES_PER_FOLDER]

        selected_paths = {
            s["path"] for s in selected
        }

        # Pasta de destino
        output_folder = OUTPUT_DIR
        output_folder.mkdir(parents=True, exist_ok=True)
        

        print(f"\n\nPasta: {folder}")
        print(f"Candidatos válidos: {len(valid_candidates)}")

        print("\nSPRITES SELECIONADAS:")

        for index, sprite in enumerate(selected, start=1):

            source = sprite["path"]

            destination = output_folder / f"{folder.name}_{source.name}"

            shutil.copy2(source, destination)

            selected_total += 1

            print(
                f"  {index}. {source.name} | "
                f"{sprite['width']}x{sprite['height']} | "
                f"{sprite['opaque_pixels']} pixels"
            )

        # Registra todas as imagens no relatório
        for sprite in candidates:

            is_selected = sprite["path"] in selected_paths

            if not is_selected:
                rejected_total += 1

            report.append({
                "folder": str(folder),
                "filename": sprite["path"].name,
                "width": sprite["width"],
                "height": sprite["height"],
                "opaque_pixels": sprite["opaque_pixels"],
                "bbox_area": sprite["bbox_area"],
                "density": round(sprite["density"], 4),
                "valid": sprite["valid"],
                "selected": is_selected
            })

    # ======================================
    # RELATÓRIO CSV
    # ======================================

    report_path = OUTPUT_DIR / "selection_report.csv"

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        if report:

            writer = csv.DictWriter(
                file,
                fieldnames=report[0].keys()
            )

            writer.writeheader()
            writer.writerows(report)

    # ======================================
    # RESULTADO
    # ======================================

    print("\n================================")
    print("SELEÇÃO CONCLUÍDA")
    print("================================")

    print(f"Personagens/pastas: {len(folders)}")
    print(f"Sprites selecionadas: {selected_total}")
    print(f"Sprites descartadas: {rejected_total}")
    print(f"Máximo por pasta: {IMAGES_PER_FOLDER}")
    print(f"Destino: {OUTPUT_DIR.resolve()}")
    print(f"Relatório: {report_path.resolve()}")


if __name__ == "__main__":
    main()
