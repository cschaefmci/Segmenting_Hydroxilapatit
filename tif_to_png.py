from pathlib import Path
from PIL import Image, ImageSequence

def save_tif_as_png(in_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_paths = list(in_dir.rglob("*.tif")) + list(in_dir.rglob("*.tiff"))

    for tif_path in tif_paths:
        with Image.open(tif_path) as im:
            n = getattr(im, "n_frames", 1)

            if n == 1:
                frame = im if im.mode in ("L", "I;16", "RGB", "RGBA") else im.convert("RGB")
                out_path = out_dir / f"{tif_path.stem}.png"
                frame.save(out_path)
            else:
                for i, frame in enumerate(ImageSequence.Iterator(im)):
                    frame = frame.copy()
                    if frame.mode not in ("L", "I;16", "RGB", "RGBA"):
                        frame = frame.convert("RGB")
                    out_path = out_dir / f"{tif_path.stem}_p{i:02d}.png"
                    frame.save(out_path)

# Eingabe und Ausgabeordner festlegen
IN_DIR  = Path(r"C:\Users\Constantin\Pictures\BA\2181\1_3\picture")
OUT_DIR = Path(r"C:\Users\Constantin\Pictures\BA\2181")

# Konvertierung starten
save_tif_as_png(IN_DIR, OUT_DIR)