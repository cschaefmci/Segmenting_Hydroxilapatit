import os
import time
import glob
import json
import math
from typing import Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt

from threshold_functions import (
    load_image,
    load_and_invert_mask,
    apply_mask,
    HA_thresh_otsu,
    close_touching_shifted,
    find_biggest_contour,
    display_images
)

# -------------------------------
# Projektpfade
# -------------------------------

project_root = r"C:\Users\Constantin\Desktop\Segmenting_Hydroxilapatit"

images_dir = os.path.join(project_root, r"input_files\images\test")
tps_dir    = os.path.join(project_root, r"input_files\TPS_layer\test")
gt_dir     = os.path.join(project_root, r"input_files\masks\test")

out_root   = os.path.join(project_root, r"Test_Results\otsu_threshold_runs")
os.makedirs(out_root, exist_ok=True)

save_viz = True

# Parameter des Otsu Ansatzes
band_halfwidth = 20          # Bandbreite um den Otsu Schwellwert
shift_dy       = 8           # Verschiebung der TPS Kontur nach oben
close_ksize    = 8           # Kernelgröße für das Schließen


# -------------------------------
# Hilfsfunktionen für Metriken
# -------------------------------

def basename_noext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def ensure_u8_binary(a: np.ndarray) -> np.ndarray:
    """Sichert ein Binärbild mit Werten 0 und 255."""
    return ((a > 0).astype(np.uint8) * 255)

def thickness_per_column(mask_u8: np.ndarray) -> np.ndarray:
    """Zählt Vordergrundpixel je Spalte."""
    return (mask_u8 > 0).sum(axis=0).astype(np.int32)

def dice_iou(pred_u8: np.ndarray, gt_u8: np.ndarray) -> Tuple[float, float]:
    """Berechnet Dice und IoU basierend auf 0 und 255."""
    p = (pred_u8 > 0).astype(np.uint8)
    g = (gt_u8   > 0).astype(np.uint8)
    inter = int(np.sum((p & g) > 0))
    pp    = int(np.sum(p > 0))
    gg    = int(np.sum(g > 0))
    union = pp + gg - inter
    dice = (2.0 * inter / (pp + gg)) if (pp + gg) > 0 else 0.0
    iou  = (inter / union) if union > 0 else 0.0
    return float(dice), float(iou)

def coverage_recall(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    """Abdeckung der GT durch die Vorhersage."""
    p = (pred_u8 > 0).astype(np.uint8)
    g = (gt_u8   > 0).astype(np.uint8)
    gt_pos = int(np.sum(g > 0))
    if gt_pos == 0:
        return float("nan")
    hit = int(np.sum((p & g) > 0))
    return float(hit / gt_pos)

def otsu_threshold_value(img_u8: np.ndarray) -> int:
    """Gibt den Otsu Schwellwert für ein Graubild zurück."""
    thr, _ = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(round(thr))

def pseudo_prob_from_otsu(img_u8: np.ndarray, thr: int, halfwidth: int) -> np.ndarray:
    """
    Pseudowahrscheinlichkeit basierend auf der Distanz zum Otsu Schwellwert.
    Werte innerhalb des Bandes erhalten hohe Werte.
    Rückgabe float in [0,1].
    """
    dist = np.abs(img_u8.astype(np.int16) - thr)
    p = 1.0 - (dist / float(max(1, halfwidth)))
    p = np.clip(p, 0.0, 1.0)
    return p.astype(np.float32)


# -------------------------------
# Paarbildung
# -------------------------------

img_exts = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")

imgs = []
for ext in img_exts:
    imgs += glob.glob(os.path.join(images_dir, ext))
imgs = sorted(imgs)

tps = sorted(glob.glob(os.path.join(tps_dir, "*.png")))
gts = sorted(glob.glob(os.path.join(gt_dir, "*.png")))

img_map = {basename_noext(p): p for p in imgs}
tps_map = {basename_noext(p): p for p in tps}
gt_map  = {basename_noext(p): p for p in gts}

common_keys = sorted(set(img_map) & set(tps_map) & set(gt_map))
pairs = [(img_map[k], tps_map[k], gt_map[k]) for k in common_keys]

print(f"Gefundene Tripel: {len(pairs)}")
if not pairs:
    raise RuntimeError("Keine passenden Bild Tripel gefunden")


# -------------------------------
# Ausgabeordner anlegen
# -------------------------------

prob_dir      = os.path.join(out_root, "prob_maps")
pred_dir      = os.path.join(out_root, "pred_masks")
gt_out_dir    = os.path.join(out_root, "gt_mask")
th_pred_dir   = os.path.join(out_root, "thickness_pred")
th_gt_dir     = os.path.join(out_root, "thickness_gt")
metrics_dir   = os.path.join(out_root, "metrics")
prompts_dir   = os.path.join(out_root, "prompts")
metadata_dir  = os.path.join(out_root, "metadata")
viz_dir       = os.path.join(out_root, "viz")

for d in [prob_dir, pred_dir, gt_out_dir, th_pred_dir, th_gt_dir, metrics_dir, prompts_dir, metadata_dir, viz_dir]:
    os.makedirs(d, exist_ok=True)


# -------------------------------
# Verarbeitung
# -------------------------------

for idx, (img_path, tps_path, gt_path) in enumerate(pairs, 1):
    stem = basename_noext(img_path)
    t0 = time.perf_counter()
    print(f"[{idx}/{len(pairs)}] {stem}")

    try:
        # 1. Eingang laden
        original = load_image(img_path)
        tps_mask_inv, contour_TPS = load_and_invert_mask(tps_path)

        # 2. Auf TPS begrenzen
        masked_orig = apply_mask(original, tps_mask_inv, mode="and")

        # 3. Otsu Band und binäre Kandidaten
        #    HA_thresh_otsu liefert ein Binärbild auf Basis des Bandes ± band_halfwidth
        #    Für die Pseudowahrscheinlichkeit wird der Otsu Schwellwert separat bestimmt
        ha_candidates = HA_thresh_otsu(masked_orig)
        thr_val = otsu_threshold_value(masked_orig)
        prob_float = pseudo_prob_from_otsu(masked_orig, thr_val, band_halfwidth)

        # 4. Verbinden der nahe an TPS liegenden Komponenten
        ha_closed = close_touching_shifted(
            ha_candidates,
            contour_TPS,
            dy=shift_dy,
            ksize=close_ksize
        )

        # 5. Größte zusammenhängende Struktur wählen und füllen
        _, contour_HA = find_biggest_contour(ha_closed)
        if contour_HA is not None:
            HA = np.zeros_like(original, dtype=np.uint8)
            cv2.drawContours(HA, [contour_HA], -1, 255, cv2.FILLED)
        else:
            HA = np.zeros_like(original, dtype=np.uint8)

        # 6. Referenz laden und auf TPS begrenzen
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(f"GT nicht gefunden: {gt_path}")
        gt_bin = ensure_u8_binary(gt)
        gt_masked = cv2.bitwise_and(gt_bin, tps_mask_inv)

        # 7. Artefakte schreiben
        prob_u8 = np.rint(np.clip(prob_float * 255.0, 0, 255)).astype(np.uint8)
        out_prob = os.path.join(prob_dir, f"{stem}_prob.png")
        out_pred = os.path.join(pred_dir, f"{stem}_pred.png")
        out_gt   = os.path.join(gt_out_dir, f"{stem}_gt.png")

        cv2.imwrite(out_prob, prob_u8)
        cv2.imwrite(out_pred, HA)
        cv2.imwrite(out_gt,   gt_masked)

        # 8. Dickenprofile
        th_pred = thickness_per_column(HA)
        th_gt   = thickness_per_column(gt_masked)
        W_orig  = original.shape[1]
        idxs    = np.arange(W_orig, dtype=np.int32)

        out_th_pred = os.path.join(th_pred_dir, f"{stem}_thickness_pred.csv")
        out_th_gt   = os.path.join(th_gt_dir,   f"{stem}_thickness_gt.csv")

        np.savetxt(out_th_pred,
                   np.c_[idxs, th_pred],
                   delimiter=",",
                   header="spaltenindex,dicke_pixel",
                   fmt="%d", comments="")
        np.savetxt(out_th_gt,
                   np.c_[idxs, th_gt],
                   delimiter=",",
                   header="spaltenindex,dicke_pixel",
                   fmt="%d", comments="")

        # 9. Bildmetriken
        dice, iou = dice_iou(HA, gt_masked)
        diff = th_pred.astype(np.int32) - th_gt.astype(np.int32)
        mae  = float(np.mean(np.abs(diff))) if diff.size else 0.0
        bias = float(np.mean(diff)) if diff.size else 0.0
        mabs = int(np.max(np.abs(diff))) if diff.size else 0
        cover = coverage_recall(HA, gt_masked)
        t_ms = (time.perf_counter() - t0) * 1000.0

        metrics = {
            "dice": dice,
            "iou": iou,
            "mae_dicke": mae,
            "bias_dicke": bias,
            "max_abs_err": mabs,
            "abdeckung": cover,
            "laufzeit_ms": float(round(t_ms, 1))
        }
        with open(os.path.join(metrics_dir, f"{stem}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # 10. Protokoll für die Schwellenmethode
        prompt_log = {
            "methode": "otsu_band",
            "parameter": {
                "band_halfwidth": int(band_halfwidth),
                "shift_dy": int(shift_dy),
                "close_ksize": int(close_ksize),
                "otsu_threshold": int(thr_val)
            }
        }
        with open(os.path.join(prompts_dir, f"{stem}_prompt_protokoll.json"), "w", encoding="utf-8") as f:
            json.dump(prompt_log, f, indent=2, ensure_ascii=False)

        # 11. Metadaten
        metadata = {
            "bildkennung": stem,
            "eingabe_bild": img_path,
            "eingabe_tps": tps_path,
            "eingabe_gt": gt_path,
            "verfahrensart": "Otsu Band um Schwellwert",
            "prob_map_definition": "Nahewert zu Otsu Schwellwert im Band",
            "band_halfwidth": int(band_halfwidth),
            "shift_dy": int(shift_dy),
            "close_ksize": int(close_ksize)
        }
        with open(os.path.join(metadata_dir, f"{stem}_bildmetadaten.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 12. Visualisierung
        if save_viz:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(masked_orig, cmap="gray"); axes[0].set_title("Bild ohne TPS"); axes[0].axis("off")
            axes[1].imshow(gt_masked,  cmap="gray");  axes[1].set_title("Referenz"); axes[1].axis("off")
            axes[2].imshow(prob_u8, vmin=0, vmax=255); axes[2].set_title("Pseudowahrscheinlichkeit"); axes[2].axis("off")
            axes[3].imshow(HA, cmap="gray"); axes[3].set_title("Vorhersage Otsu"); axes[3].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{stem}_viz.png"), dpi=150)
            plt.close(fig)

        print(f"{stem}: prob, pred, gt, Dickenprofile, Metriken, Protokoll und Metadaten gespeichert")

    except Exception as e:
        print(f"Fehler bei {stem}: {e}")