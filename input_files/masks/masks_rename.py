import os
import glob
import re

# Ordner anpassen
img_dir = r"C:\Users\Constantin\Bilder\2303\4\pictures"
mask_dir = r"C:\Users\Constantin\Bilder\2303\4\mask_HA"

image_paths = glob.glob(os.path.join(img_dir, "*.tif"))
mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))

# Alle Masken als dict mit reduziertem Key (z. B. '2118_1_1 (1)')
mask_map = {}
for m in mask_paths:
    base = os.path.splitext(os.path.basename(m))[0]
    # Entferne '_mask' wenn vorhanden
    key = base.replace("_mask", "").strip()
    mask_map[key] = m

# Jetzt Bilder durchgehen
for img_path in image_paths:
    img_name = os.path.splitext(os.path.basename(img_path))[0]  # z.B. '2118_1_1 (1)'
    target_mask_name = img_name + ".png"
    
    if img_name in mask_map:
        original_mask = mask_map[img_name]
        target_path = os.path.join(mask_dir, target_mask_name)
        os.rename(original_mask, target_path)
        print(f"✅ Umbenannt: {os.path.basename(original_mask)} → {target_mask_name}")
    else:
        print(f"⚠️ Keine passende Maske für Bild: {img_name}")