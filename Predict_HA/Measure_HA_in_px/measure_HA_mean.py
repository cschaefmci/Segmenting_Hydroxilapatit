import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple

def height_profile_largest_white_component(img: np.ndarray, connectivity: int = 8) -> List[int]:
    if img.ndim != 2:
        raise ValueError("Erwartet ein zweidimensionales Graustufenbild")

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    if not np.any(bin_img):
        return [0] * bin_img.shape[1]

    bin01 = (bin_img > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=connectivity)

    if num_labels <= 1:
        return [0] * bin_img.shape[1]

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    largest_mask = (labels == largest_label)
    heights = largest_mask.sum(axis=0).astype(int)
    return heights.tolist()

def mean_height_per_image(img: np.ndarray) -> float:
    heights = height_profile_largest_white_component(img, connectivity=8)
    return float(np.mean(heights)) if heights else 0.0

def process_folder(folder: str) -> Tuple[List[Tuple[str, float]], float]:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Verzeichnis nicht gefunden: {folder}")

    results: List[Tuple[str, float]] = []
    for png_path in sorted(folder_path.glob("*.png")):
        img_gray = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Konnte Bild nicht laden: {png_path.name}")
            continue
        m = mean_height_per_image(img_gray)
        results.append((png_path.name, m))
        print(f"{png_path.name}: {m:.2f} Pixel")

    overall_mean = float(np.mean([m for _, m in results])) if results else 0.0
    if results:
        print(f"Gesamtmittel über {len(results)} Bilder: {overall_mean:.2f} Pixel")
    else:
        print("Keine PNG Dateien gefunden")

    return results, overall_mean
  

if __name__ == "__main__":
    # Pfad anpassen
    ordner = r"C:\Users\Constantin\Desktop\Segmenting_Hydroxilapatit\input_files\masks\train"
    process_folder(ordner)