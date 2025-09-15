import os
import glob
import tifffile
import patchify
from patchify import unpatchify
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import Image
import torch
from transformers import SamModel, SamConfig, SamProcessor
from datasets import Dataset as HFDataset

def invert_mask(path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
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

def shift_mask_binary(m: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Verschiebt eine binäre Maske um dx und dy in Pixeln.
    dy kleiner null bedeutet Verschiebung nach oben.
    Rückgabe hat denselben Typ und dieselbe Form.
    """
    h, w = m.shape
    out = np.zeros_like(m)
    # Zielbereich im Ausgabebild
    y0 = max(0,  dy);  y1 = min(h, h + dy)
    x0 = max(0,  dx);  x1 = min(w, w + dx)
    # Quellbereich im Eingabebild
    yy0 = max(0, -dy); yy1 = yy0 + (y1 - y0)
    xx0 = max(0, -dx); xx1 = xx0 + (x1 - x0)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = m[yy0:yy1, xx0:xx1]
    return out

# um keine Bildinformationen zu verlieren muss die Höhe auf ein Vielfaches von 256 erweitert werden
def pad_to_multiple(a: np.ndarray, multiple: int) -> np.ndarray:
    H, W = a.shape[:2]
    padH = (multiple - (H % multiple)) % multiple
    padW = (multiple - (W % multiple)) % multiple
    if padH or padW:
        pads = ((0, padH), (0, padW)) + ((0, 0),) * (a.ndim - 2)
        a = np.pad(a, pads, mode="constant", constant_values=0)
    return a
  
def mask_to_box(mask: np.ndarray, margin: int = 4, min_pixels: int = 1):
  assert mask.ndim == 2
  H, W = mask.shape
  ys, xs = np.where(mask > 0)
  if ys.size < min_pixels:
      return None
  y0, y1 = int(ys.min()), int(ys.max())
  x0, x1 = int(xs.min()), int(xs.max())
  if margin > 0:
      y0 = max(0, y0 - margin)
      x0 = max(0, x0 - margin)
      y1 = min(H - 1, y1 + margin)
      x1 = min(W - 1, x1 + margin)
  return float(x0), float(y0), float(x1), float(y1)

def ceil_to_multiple(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m

def make_cosine_window(p: int, eps: float = 1e-3) -> np.ndarray:
    """
    Zweidimensionales Kosinusfenster als Gewicht für weiches Blending.
    eps verhindert exakt null am Rand  damit auch Randbereiche ohne Überlappung Gewicht erhalten.
    """
    w1 = np.hanning(p).astype(np.float32)
    w2 = np.outer(w1, w1).astype(np.float32)
    return np.maximum(w2, eps)

def reconstruct_overlapping(patches: np.ndarray,
                            coords: np.ndarray,
                            H_pad: int,
                            W_pad: int,
                            patch_size: int,
                            window: np.ndarray) -> np.ndarray:
    """
    patches erwartet Form N p p  coords erwartet Form N 2 mit (y0 x0)
    Ergebnis ist ein gepaddetes Vollbild in float32
    """
    acc  = np.zeros((H_pad, W_pad), dtype=np.float32)
    wsum = np.zeros((H_pad, W_pad), dtype=np.float32)
    for patch, (y0, x0) in zip(patches, coords):
        y1 = y0 + patch_size
        x1 = x0 + patch_size
        acc[y0:y1, x0:x1]  += patch.astype(np.float32) * window
        wsum[y0:y1, x0:x1] += window
    return acc / np.maximum(wsum, 1e-6)