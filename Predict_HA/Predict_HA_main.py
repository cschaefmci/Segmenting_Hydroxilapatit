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
from Predict_HA_functions import invert_mask, apply_mask, shift_mask_binary, pad_to_multiple, mask_to_box, ceil_to_multiple, make_cosine_window, reconstruct_overlapping


# Ordner dieser .py-Datei: ...\Segmenting_Hydroxilapatit\Predict_HA
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Projekt-Root: ein Verzeichnis nach oben
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

"""
  Modell laden
  """
  
# Load the model configuration
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "Predict_HA", "SAM_models", "sam_finetuned_jitterBoxes_DiceCE_20ep_lr_e5.pth")
# Create an instance of the model architecture with the loaded configuration
my_HA_model = SamModel(config=model_config)
#Update the model by loading the weights from saved file.

#my_HA_model.load_state_dict(torch.load("/models/sam_finetuned_40ep_lr_e4.pth")) 
my_HA_model.load_state_dict((torch.load(CHECKPOINT_PATH)), strict=False) # der beste bisher: sam_finetuned_jitterBoxes_DiceCE_20ep_lr_e5.pth

# set the device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

my_HA_model.to(device)
my_HA_model.eval()



"""Ordner definieren
  """
IMG_DIR = os.path.join(PROJECT_ROOT, "input_files", "large_images")
IMG_PATHS = sorted(glob.glob(os.path.join(IMG_DIR, "*.tif"))) # catch all .tif files in that directory

BIN_IMG_DIR = os.path.join(PROJECT_ROOT, "input_files", "large_bi_masks")
BIN_IMG_PATHS = sorted(glob.glob(os.path.join(BIN_IMG_DIR, "*.png"))) # catch all .png files in that directory


large_test_image = tifffile.imread(IMG_PATHS[2]) # example: load second image (index 1)
# 2303_2 (4) entspricht 3
# 2303_6 (4) entspricht 4

mask, contour_TPS = invert_mask (BIN_IMG_PATHS[2]) # contour_TPS is die TPS Kante
# 2. TPS Schicht ausschneiden (GW=0)
masked_orig        = apply_mask(large_test_image, mask, mode='and')

# Nullregion als Binärbild
zero_mask = (masked_orig == 0).astype(np.uint8) * 255   # Werte 0 und 255 für Morphologie

# kleine Artefakte entfernen
kernel = np.ones((3, 3), np.uint8)
zero_mask = cv2.morphologyEx(zero_mask, cv2.MORPH_OPEN, kernel)

# für die weitere Rechnung auf 0 und 1 normieren
m_orig = (zero_mask > 0).astype(np.uint8)

# gewünschte Verschiebung
p = 110                                # Anzahl Pixel nach oben
m_shift = shift_mask_binary(m_orig, dx=0, dy=-p)   # negativ ist nach oben

# gerichteter Streifen nur neu belegte Pixel
HA = ((m_shift == 1) & (m_orig == 0)).astype(np.uint8)    # Werte 0 und 1

# Bild zerschneiden

patch_size = 256
step = 128

assert masked_orig.ndim == 2 and HA.ndim == 2 and masked_orig.dtype == np.uint8 and HA.dtype == np.uint8
assert masked_orig.shape == HA.shape

# auf Patchraster padden
img = pad_to_multiple(masked_orig, patch_size)
msk = pad_to_multiple(HA, patch_size)

# Patchify
img_p = patchify.patchify(img, (patch_size, patch_size), step=step)
msk_p = patchify.patchify(msk, (patch_size, patch_size), step=step)

img_patches, mask_patches, coords = [], [], []
for i in range(img_p.shape[0]):
    for j in range(img_p.shape[1]):
        y0 = i * step
        x0 = j * step
        img_patches.append(img_p[i, j])                           # 256 x 256
        mask_patches.append(msk_p[i, j].astype(np.uint8)) # 0 oder 1 gibt an ob das Bild einen teil der Maske enthält oder nicht
        coords.append((y0, x0))

img_patches = np.array(img_patches)    # N x 256 x 256
mask_patches = np.array(mask_patches)  # N x 256 x 256
coords = np.array(coords, dtype=np.int32)              # N x 2



# img_patches und mask_patches: Form N, 256, 256
N, H, W = img_patches.shape
probs = np.zeros((N, H, W), dtype=np.float32)
preds = np.zeros((N, H, W), dtype=np.uint8)
used_boxes = [None] * N

my_HA_model.eval()
with torch.no_grad():
  for k in range(N):
      img_patch = img_patches[k]          # uint8
      msk_patch = mask_patches[k]         # 0 oder 255

      box = mask_to_box(msk_patch, margin=4, min_pixels=1)
      pil_img = Image.fromarray(img_patch).convert("RGB")

      if box is None:
          inputs = processor(images=pil_img, return_tensors="pt")
      else:
          x0, y0, x1, y1 = box
          inputs = processor(images=pil_img, input_boxes=[[[x0, y0, x1, y1]]], return_tensors="pt")
          used_boxes[k] = (x0, y0, x1, y1)

      inputs = {kk: vv.to(device) for kk, vv in inputs.items()}

      with torch.no_grad():
          out = my_HA_model(**inputs, multimask_output=False)

      p = torch.sigmoid(out.pred_masks.squeeze(1)).cpu().numpy().squeeze()  # 256, 256
      probs[k] = p
      preds[k] = (p > 0.3).astype(np.uint8)


# Zielmaße
H_orig, W_orig = masked_orig.shape
H_pad = ceil_to_multiple(H_orig, patch_size)
W_pad = ceil_to_multiple(W_orig, patch_size)

# Fenster vorbereiten
win = make_cosine_window(patch_size)

# Rekonstruktion aus Wahrscheinlichkeiten  danach Schwellen
full_prob  = reconstruct_overlapping(probs, coords, H_pad, W_pad, patch_size, win)
full_pred  = (full_prob > 0.8).astype(np.uint8) * 255

# optional  das gepatchte Eingabebild ebenfalls so zusammenfügen  nur für Darstellung nötig
full_img   = reconstruct_overlapping(img_patches.astype(np.float32), coords, H_pad, W_pad, patch_size, win)

# Auf Originalmaß zurückschneiden
full_img_c  = full_img[:H_orig, :W_orig]
full_prob_c = full_prob[:H_orig, :W_orig]
full_pred_c = full_pred[:H_orig, :W_orig]



fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(full_img_c, cmap="gray"); axes[0].set_title("Bild")
axes[1].imshow(full_prob_c);             axes[1].set_title("Wahrscheinlichkeit")
axes[2].imshow(full_pred_c, cmap="gray");axes[2].set_title("Vorhersage")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout(); plt.show()

# Save full_pred_c as a PNG file
output_filename = "full_pred_c.png"
cv2.imwrite(output_filename, full_pred_c)
print(f"Saved {output_filename}")