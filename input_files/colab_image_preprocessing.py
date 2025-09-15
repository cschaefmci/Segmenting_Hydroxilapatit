import os
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image


def find_by_basename(base_dir: str,
                     base_name: str,
                     exts: Tuple[str, ...] = (".png", ".tif", ".tiff", ".jpg", ".jpeg")) -> Optional[str]:
    """
    Sucht eine Datei mit identischem Basisnamen in base_dir.
    """
    for ext in exts:
        p = os.path.join(base_dir, base_name + ext)
        if os.path.exists(p):
            return p
    return None


def load_grayscale_pil(path: str) -> np.ndarray:
    """
    Lädt ein Bild im Grauwertformat mit PIL und gibt ein Array im Bereich null bis zweihundertfuenfundfuenfzig aus.
    """
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def load_grayscale_cv(path: str) -> np.ndarray:
    """
    Lädt ein Bild im Grauwertformat mit OpenCV. Eignet sich fuer das Einlesen der Binärmaske.
    """
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Konnte Bild nicht laden: {path}")
    return arr


def binarize_0_255(arr: np.ndarray, thresh: int = 128) -> np.ndarray:
    """
    Schwellwertbildung auf Werte null und zweihundertfuenfundfuenfzig.
    """
    _, b = cv2.threshold(arr, thresh, 255, cv2.THRESH_BINARY)
    return b


def ensure_same_shape(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    """
    Passt die Form von other an die Form von ref an. Es wird naechster Nachbar verwendet.
    """
    if ref.shape == other.shape:
        return other
    h, w = ref.shape[:2]
    return cv2.resize(other, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_inverted_binary_mask_to_image(img: np.ndarray, bin_mask: np.ndarray) -> np.ndarray:
    """
    Erwartet eine Binärmaske, in der die Titanschicht weiss ist.
    Invertiert die Maske und setzt die entsprechenden Bildbereiche auf null.
    """
    bm = binarize_0_255(bin_mask)
    inv = cv2.bitwise_not(bm)
    res = cv2.bitwise_and(img, img, mask=inv)
    return res