import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple

def height_profile_largest_white_component(img: np.ndarray, connectivity: int = 8) -> List[int]:
    """
    Spaltenweise Höhe der größten zusammenhängenden weißen Komponente in Pixeln.
    Erwartet ein zweidimensionales Binär oder Graustufenbild mit Werten 0 bis 255.
    """
    if img.ndim != 2:
        raise ValueError("Erwartet ein zweidimensionales Graustufenbild")

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Robust auf binär bringen
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    if not np.any(bin_img):
        return [0] * bin_img.shape[1]

    bin01 = (bin_img > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=connectivity)

    if num_labels <= 1:
        return [0] * bin_img.shape[1]

    # größte Vordergrundkomponente bestimmen
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    largest_mask = (labels == largest_label)

    # Spaltenhöhe berechnen
    heights = largest_mask.sum(axis=0).astype(int)
    return heights.tolist()

def stats_per_image(img: np.ndarray) -> Tuple[float, int]:
    """Gibt Mittelwert und Maximum der Spaltenhöhen zurück."""
    heights = height_profile_largest_white_component(img, connectivity=8)
    if not heights:
        return 0.0, 0
    mean_h = float(np.mean(heights))
    max_h = int(np.max(heights))
    return mean_h, max_h

def process_folder(folder: str) -> Tuple[List[Tuple[str, float, int]], float, float]:
    """
    Verarbeitet alle PNG Dateien eines Ordners.
    Gibt je Bild Mittelwert und Maximum der Spaltenhöhen aus.
    Zusätzlich werden der Mittelwert der Mittelwerte und der Mittelwert der Maxima über alle Bilder berechnet.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Verzeichnis nicht gefunden: {folder}")

    results: List[Tuple[str, float, int]] = []
    means: List[float] = []
    maxes: List[int] = []

    for png_path in sorted(folder_path.glob("*.png")):
        img_gray = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Konnte Bild nicht laden: {png_path.name}")
            continue
        mean_h, max_h = stats_per_image(img_gray)
        results.append((png_path.name, mean_h, max_h))
        means.append(mean_h)
        maxes.append(max_h)
        print(f"{png_path.name}: Mittelwert {mean_h:.2f} Pixel, Maximum {max_h} Pixel")

    if not results:
        print("Keine PNG Dateien gefunden")
        return results, 0.0, 0.0

    avg_of_means = float(np.mean(means))
    avg_of_maxes = float(np.mean(maxes))

    print(f"Gesamtmittel der Mittelwerte über {len(results)} Bilder: {avg_of_means:.2f} Pixel")
    print(f"Gesamtmittel der Maxima über {len(results)} Bilder: {avg_of_maxes:.2f} Pixel")

    return results, avg_of_means, avg_of_maxes

if __name__ == "__main__":
    ordner = r"C:\Users\Constantin\Desktop\Segmenting_Hydroxilapatit\input_files\masks\train"
    _, avg_mean, avg_max = process_folder(ordner)