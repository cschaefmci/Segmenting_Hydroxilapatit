import os, glob
import numpy as np
from PIL import Image
from patchify import patchify

# take the split-up data and create arrays
image_dir = r"C:\Users\Constantin\Documents\MCI\Bachelor\SAM\images\val"
mask_dir = r"C:\Users\Constantin\Documents\MCI\Bachelor\SAM\masks\val"

image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))

images, masks = [], []

for img_path in image_paths:
    filename = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(mask_dir, filename + ".png")
    if os.path.exists(mask_path):
        img = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        images.append(img)
        masks.append(mask)

images = np.array(images)
masks = np.array(masks)

# patchify, so SAM can process the images in its best way possible
patch_size = 256
step = 256
img_patches, mask_patches = [], []

for i in range(images.shape[0]):
    img_p = patchify(images[i], (patch_size, patch_size), step=step)
    msk_p = patchify(masks[i], (patch_size, patch_size), step=step)
    for x in range(img_p.shape[0]):
        for y in range(img_p.shape[1]):
            img_patches.append(img_p[x, y])
            mask_patch = (msk_p[x, y] / 255).astype(np.uint8)
            mask_patches.append(mask_patch)

# filter empty masks/images - slow training down
img_patches = np.array(img_patches)
mask_patches = np.array(mask_patches)
valid = [i for i, m in enumerate(mask_patches) if m.max() != 0]
filtered_val_images = img_patches[valid]
filtered_val_masks = mask_patches[valid]

# save as .npy stacks 
np.save("val_images_filtered.npy", filtered_val_images)
np.save("val_masks_filtered.npy", filtered_val_masks)

print(f"Gespeichert: {filtered_val_images.shape} Bilder, {filtered_val_masks.shape} Masken")  # masks and images should be the same shape
