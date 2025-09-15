import sys
from pathlib import Path
import numpy as np
from PIL import Image

IN_DIR_DEFAULT = r"C:\Users\Constantin\Pictures\BA\2303\4\mask_HA"
OUT_DIR_DEFAULT = r"C:\Users\Constantin\Pictures\BA\2303\4\mask"

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def is_binary(arr: np.ndarray) -> bool:
    u = np.unique(arr)
    return u.size <= 2 and np.all(np.isin(u, [0, 255]))

def to_binary(arr: np.ndarray, thresh: int = 128) -> np.ndarray:
    return np.where(arr >= thresh, 255, 0).astype(np.uint8)

def load_gray(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)

def save_gray(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(out_path)

def process_file(p: Path, in_root: Path, out_root: Path) -> tuple[str, str]:
    arr = load_gray(p)
    if is_binary(arr):
        out_path = out_root / p.relative_to(in_root)
        save_gray(arr, out_path)
        return str(p), "bereits binär"
    bin_arr = to_binary(arr)
    if not is_binary(bin_arr):
        return str(p), "Auffälligkeit nach Konversion"
    out_path = out_root / p.relative_to(in_root)
    save_gray(bin_arr, out_path)
    return str(p), "konvertiert"

def main():
    in_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(IN_DIR_DEFAULT)
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(OUT_DIR_DEFAULT)

    files = [p for p in in_dir.rglob("*") if p.suffix.lower() in EXTS]
    if not files:
        print("Keine passenden Bilddateien gefunden")
        return

    n_ok = n_conv = n_bad = 0
    for p in files:
        try:
            _, status = process_file(p, in_dir, out_dir)
            if status == "bereits binär":
                n_ok += 1
            elif status == "konvertiert":
                n_conv += 1
            else:
                n_bad += 1
                print(f"Auffällig: {p}")
        except Exception as e:
            n_bad += 1
            print(f"Fehler bei {p}: {e}")

    print("Zusammenfassung")
    print(f"bereits binär    {n_ok}")
    print(f"konvertiert      {n_conv}")
    print(f"auffällig        {n_bad}")

if __name__ == "__main__":
    main()