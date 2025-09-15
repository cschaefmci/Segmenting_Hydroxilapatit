import os
import glob
from typing import List
import numpy as np
from patchify import patchify
from pathlib import Path

from colab_image_preprocessing import ( find_by_basename, load_grayscale_pil, load_grayscale_cv, binarize_0_255, ensure_same_shape, apply_inverted_binary_mask_to_image,)

# Eingabepfade
image_dir = r"C:\Users\Constantin\Documents\MCI\Bachelor\SAM\images\train"
mask_dir = r"C:\Users\Constantin\Documents\MCI\Bachelor\SAM\masks\train"
binary_image_dir = r"C:\Users\Constantin\Documents\MCI\Bachelor\SAM\binary_images\train"

# Patchparameter
patch_size = 256
step = 256

sam_root = Path(image_dir).parents[1]
out_dir = sam_root / "patched_images"
out_dir.mkdir(parents=True, exist_ok=True)

out_images = str(out_dir / "train_images_filtered_blacked.npy")
out_masks = str(out_dir / "train_masks_filtered_blacked.npy")


def main() -> None:
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    if not image_paths:
        print("Keine Eingabebilder gefunden")
        return

    img_patches: List[np.ndarray] = []
    mask_patches: List[np.ndarray] = []

    used = 0
    skipped = 0

    for img_path in image_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]

        gt_mask_path = find_by_basename(mask_dir, base)
        bin_mask_path = find_by_basename(binary_image_dir, base)

        if gt_mask_path is None or bin_mask_path is None:
            skipped += 1
            continue

        # Laden
        img = load_grayscale_pil(img_path)
        gt_mask = load_grayscale_pil(gt_mask_path)
        bin_mask = load_grayscale_cv(bin_mask_path)

        # Form abgleichen
        bin_mask = ensure_same_shape(img, bin_mask)

        # Invertierte Binärmaske auf Bild anwenden
        img_masked = apply_inverted_binary_mask_to_image(img, bin_mask)

        # Ground Truth bleibt vom Maskieren unberührt, wird aber binarisiert
        gt_mask_bin = binarize_0_255(gt_mask)

        # Patchen
        ip = patchify(img_masked, (patch_size, patch_size), step=step)
        mp = patchify(gt_mask_bin, (patch_size, patch_size), step=step)

        # Sammeln mit Filter auf nicht leere Ground Truth Patches
        for x in range(ip.shape[0]):
            for y in range(ip.shape[1]):
                p_img = ip[x, y]
                p_msk = mp[x, y]
                p_msk01 = (p_msk // 255).astype(np.uint8)
                if p_msk01.max() == 0:
                    continue
                img_patches.append(p_img)
                mask_patches.append(p_msk01)

        used += 1

    if not img_patches:
        print("Keine Patches nach der Filterung vorhanden")
        return

    img_patches_arr = np.array(img_patches, dtype=np.uint8)
    mask_patches_arr = np.array(mask_patches, dtype=np.uint8)

    np.save(out_images, img_patches_arr)
    np.save(out_masks, mask_patches_arr)

    print(f"Verarbeitete Bilder: {used}, uebersprungene Bilder: {skipped}")
    print(f"Gespeichert in {out_dir}")
    print(f"Gespeichert: {img_patches_arr.shape} Bilder, {mask_patches_arr.shape} Masken")
    
if __name__ == "__main__":
    main()