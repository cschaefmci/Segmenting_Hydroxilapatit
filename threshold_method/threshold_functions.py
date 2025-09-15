import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {path}")
    return img

def load_and_invert_mask(path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Lädt ein Grauwert-Bild, findet Außenkonturen der weißen Regionen,
    erstellt eine invertierte Füll-Maske (innen=0, außen=255) und
    gibt (Maske, Konturen) zurück.
    """
    mask_orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask_orig is None:
        raise FileNotFoundError(f"Maske nicht gefunden: {path}")
    contour_TPS, _ = cv2.findContours(mask_orig,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones_like(mask_orig, dtype=np.uint8) * 255 
    cv2.drawContours(mask, contour_TPS, -1, 0, thickness=cv2.FILLED)
    return mask, contour_TPS

def apply_mask(img: np.ndarray, mask: np.ndarray, mode: str = 'and') -> np.ndarray:
    """
    mode='and': klassisches Maskieren (AND)
    mode='or' : überall mask=255 → Pixel auf 255 setzen (OR)
    """
    if mode == 'and':
        return cv2.bitwise_and(img, img, mask=mask)
    elif mode == 'or':
        return cv2.bitwise_or(img, mask)
    else:
        raise ValueError("apply_mask: mode muss 'and' oder 'or' sein")

def HA_thresh_otsu(img: np.ndarray) -> np.ndarray:
    """
    Otsu ±20 Graustufen als Graustufenberiech der HA Schicht -> Binärbild (0/255)
    """
    thresh, _ = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.inRange(img, thresh - 20, thresh + 20)
  
def shift_contours(contours: List[np.ndarray],
                   dx: int,
                   dy: int,
                   img_shape: Tuple[int,int]) -> List[np.ndarray]:
    """
    Verschiebt jede Kontur in der Liste um (dx, dy).
    dy>0 → nach unten. Clamping auf img_shape (h, w).
    """
    h, w = img_shape
    shifted = []
    for cnt in contours:
        cnt2 = cnt.copy()
        cnt2[:,0,0] = np.clip(cnt2[:,0,0] + dx, 0, w-1)
        cnt2[:,0,1] = np.clip(cnt2[:,0,1] - dy, 0, h-1)
        shifted.append(cnt2)
    return shifted

def close_touching_shifted(binary_img: np.ndarray,
                           contours: List[np.ndarray],
                           dy: int,
                           ksize: int) -> np.ndarray:
    """
    Schließt nur jene Objekte in binary_img, deren Konturen
    nach Verschiebung um dy direkt am Objekt angrenzen.
    ksize: ungerade Kernelgröße für cv2.MORPH_CLOSE.
    """
    # 1) Label map erzeugen
    _, labels = cv2.connectedComponents(binary_img)
    h, w = binary_img.shape

    # 2) Konturen um dy verschieben
    shifted = shift_contours(contours, dx=0, dy=dy, img_shape=(h, w))

    # 3) Alle Punkte der verschobenen Konturen sammeln
    ref_pts = {(pt[0], pt[1]) for cnt in shifted for pt in cnt.reshape(-1,2)}

    # 4) Labels ermitteln, die an diese Punkte angrenzen
    labels_to_close = set()
    for x, y in ref_pts:
        for dx2, dy2 in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx2, y + dy2
            if 0 <= nx < w and 0 <= ny < h:
                lbl = labels[ny, nx]
                if lbl != 0:
                    labels_to_close.add(int(lbl))

    # 5) Maske dieser Labels
    touch_mask = np.isin(labels, list(labels_to_close)).astype(np.uint8) * 255

    # 6) Morphologisches Schließen auf dieser Maske
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(touch_mask, cv2.MORPH_CLOSE, kernel)

    # 7) Nur in geschlossenen Bereichen im Original weiß setzen
    out = binary_img.copy()
    out[closed == 255] = 255
    return out
  
def find_biggest_contour(img):
    # Find contours in the binary image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the biggest contour based on area
    if contours:
        biggest_contour = max(contours, key=cv2.contourArea)
        return img, biggest_contour
    else:
        return img, None

def display_images(images: List[Tuple[np.ndarray,str,str]], cmap='gray'):
    """
    Zeigt eine Liste von (img, title, cmap)-Triples nebeneinander.
    """
    n = len(images)
    if n == 0:
        return
    plt.figure(figsize=(5 * n, 4))
    for idx, (img, title, cm) in enumerate(images):
        plt.subplot(1, n, idx + 1)
        plt.imshow(img, cmap=cm or cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()