"""
Tri-Check Anemia Triage — Data Engine
======================================
Processes three datasets (Eyes, Tongue, Nails).
Uses OpenCV for cropping (no MediaPipe dependency here — MediaPipe is only
used in app.py at inference time where it works fine via the tasks API).

Feature extraction: CLAHE  →  R/G ratio  +  LAB a* channel per ROI.
Output: tri_check_master.csv

Run: python src/data_engine.py
"""

import cv2
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
EYES_DIR   = DATA_DIR / "India_Eyes"
TONGUE_DIR = DATA_DIR / "India_Tongue"
NAILS_DIR  = DATA_DIR / "India_Nails"
OUT_CSV    = BASE_DIR / "tri_check_master.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ── OpenCV Haar cascades (used for eye / face detection) ──────────────────────
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def apply_clahe(bgr_img):
    """CLAHE on the L-channel of LAB to normalise illumination."""
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def extract_color_features(bgr_img, mask=None):
    """
    Extract rg_ratio, lab_a, mean_r, mean_g, mean_b from a BGR image.
    If a binary mask is supplied only masked pixels are used.
    """
    if mask is not None:
        pixels = bgr_img[mask > 0]
    else:
        pixels = bgr_img.reshape(-1, 3)

    if len(pixels) == 0:
        return {"rg_ratio": 0.0, "lab_a": 0.0,
                "mean_r": 0.0, "mean_g": 0.0, "mean_b": 0.0}

    b_vals = pixels[:, 0].astype(float)
    g_vals = pixels[:, 1].astype(float)
    r_vals = pixels[:, 2].astype(float)

    mean_r   = r_vals.mean()
    mean_g   = g_vals.mean()
    mean_b   = b_vals.mean()
    rg_ratio = mean_r / (mean_g + 1e-6)

    lab_roi = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    lab_pix = lab_roi[mask > 0] if mask is not None else lab_roi.reshape(-1, 3)
    lab_a   = float(lab_pix[:, 1].astype(float).mean()) if len(lab_pix) > 0 else 0.0

    return {
        "rg_ratio": round(float(rg_ratio), 6),
        "lab_a":    round(lab_a,           6),
        "mean_r":   round(float(mean_r),   4),
        "mean_g":   round(float(mean_g),   4),
        "mean_b":   round(float(mean_b),   4),
    }


def safe_imread(path):
    img = cv2.imread(str(path))
    if img is None:
        log.warning(f"  Could not read: {path.name}")
    return img


def center_crop(img, ratio=0.5):
    """Crop the central (ratio x ratio) portion of the image."""
    h, w   = img.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y1     = (h - ch) // 2
    x1     = (w - cw) // 2
    return img[y1:y1 + ch, x1:x1 + cw]


def bottom_center_crop(img, h_ratio=0.4, w_ratio=0.6):
    """
    Crop the bottom-centre region — good for conjunctiva in eye closeups
    where the lower eyelid tends to sit in the lower half of the frame.
    """
    h, w   = img.shape[:2]
    ch, cw = int(h * h_ratio), int(w * w_ratio)
    y1     = h - ch
    x1     = (w - cw) // 2
    return img[y1:y1 + ch, x1:x1 + cw]


def detect_eye_roi(bgr_img):
    """
    Try to detect an eye region with Haar cascades.
    Returns the cropped eye ROI, or falls back to bottom_center_crop.
    """
    gray  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    eyes  = _EYE_CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=4, minSize=(20, 20))
    if len(eyes) > 0:
        # Pick the largest detected eye region
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)
        x, y, w, h = eyes[0]
        # Take the lower third of the eye box — that is the conjunctiva
        y_conj = y + int(h * 0.55)
        h_conj = int(h * 0.45)
        roi    = bgr_img[y_conj:y_conj + h_conj, x:x + w]
        if roi.size > 0:
            return roi
    return bottom_center_crop(bgr_img)


# ══════════════════════════════════════════════════════════════════════════════
# Eyes Dataset
# ══════════════════════════════════════════════════════════════════════════════

def process_eyes():
    """
    India_Eyes/
        anemia_dataset.csv   Number, Hb, Anaemic (Yes/No)
        master_dataset.csv   Red_Pct, Green_Pct, LAB_A, Anemic  (pre-extracted)
        1/ 2/ 3/ ...         subfolders — Number maps to folder name

    ROI strategy: Haar eye detector -> bottom-center crop fallback.
    """
    log.info("=== Processing Eyes dataset ===")
    records = []

    if not EYES_DIR.exists():
        log.warning(f"  Eyes dir not found: {EYES_DIR}")
        return records

    # ── Load anemia_dataset.csv ────────────────────────────────────────────────
    num_to_label = {}
    anemia_csv   = EYES_DIR / "anemia_dataset.csv"
    if anemia_csv.exists():
        meta = pd.read_csv(anemia_csv)
        meta.columns = [c.strip() for c in meta.columns]
        num_col = next(
            (c for c in meta.columns if c.lower() in ("number", "patient_id", "id")), None
        )
        lbl_col = next(
            (c for c in meta.columns if c.lower() in ("anaemic", "anemic", "label")), None
        )
        if num_col and lbl_col:
            for _, row in meta.iterrows():
                try:
                    folder_num = str(int(float(row[num_col])))
                    raw        = str(row[lbl_col]).strip().lower()
                    label      = 1 if raw in ("yes", "1", "true") else 0
                    num_to_label[folder_num] = label
                except (ValueError, TypeError):
                    continue
            an = sum(num_to_label.values())
            log.info(
                f"  anemia_dataset.csv: {len(num_to_label)} patients "
                f"(anaemic={an}, healthy={len(num_to_label)-an})"
            )
        else:
            log.warning(f"  Unexpected columns: {list(meta.columns)}")
    else:
        log.warning("  anemia_dataset.csv not found.")

    # ── Walk numbered subfolders ───────────────────────────────────────────────
    subfolder_count = 0
    for subdir in sorted(EYES_DIR.iterdir(), key=lambda p: p.name):
        if not subdir.is_dir():
            continue
        folder_name = subdir.name
        label = num_to_label.get(
            folder_name,
            1 if any(k in folder_name.lower() for k in ("anaem", "anemic", "yes")) else 0,
        )

        img_files = sorted(
            p for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not img_files:
            continue
        subfolder_count += 1
        log.info(f"  Folder '{folder_name}': {len(img_files)} image(s) -> label={label}")

        for img_path in img_files:
            img = safe_imread(img_path)
            if img is None:
                continue
            img = apply_clahe(img)
            roi = detect_eye_roi(img)
            feats           = extract_color_features(roi)
            feats["label"]  = label
            feats["source"] = "eyes"
            records.append(feats)

    log.info(f"  Eyes (images): {len(records)} samples from {subfolder_count} folders.")

    # ── Ingest master_dataset.csv bonus rows ───────────────────────────────────
    master_csv = EYES_DIR / "master_dataset.csv"
    if master_csv.exists():
        mdf   = pd.read_csv(master_csv)
        mdf.columns = [c.strip() for c in mdf.columns]
        bonus = 0
        for _, row in mdf.iterrows():
            try:
                red_pct   = float(row.get("Red_Pct",   0))
                green_pct = float(row.get("Green_Pct", 0))
                lab_a     = float(row.get("LAB_A",     128))
                label     = int(row.get("Anemic",      0))
                mean_r    = red_pct   * 2.55
                mean_g    = green_pct * 2.55
                mean_b    = max(255 - mean_r - mean_g, 0)
                rg_ratio  = mean_r / (mean_g + 1e-6)
                records.append({
                    "rg_ratio": round(rg_ratio, 6),
                    "lab_a":    round(lab_a,    6),
                    "mean_r":   round(mean_r,   4),
                    "mean_g":   round(mean_g,   4),
                    "mean_b":   round(mean_b,   4),
                    "label":    label,
                    "source":   "eyes",
                })
                bonus += 1
            except (ValueError, TypeError):
                continue
        log.info(f"  Eyes (CSV bonus): {bonus} rows from master_dataset.csv.")
    else:
        log.info("  master_dataset.csv not found — skipping bonus rows.")

    log.info(f"  Eyes TOTAL: {len(records)} samples.")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Tongue Dataset
# ══════════════════════════════════════════════════════════════════════════════

def process_tongue():
    """
    India_Tongue/
        dataset/        1.bmp, 2.bmp ...
        groundtruth/
            images/     (not used)
            mask/       1.png, 2.png ... (binary masks, matched by stem)

    Mask found  -> feature extraction from masked pixels.
    No mask     -> center crop of image.
    Proxy label: LAB a* < 145 -> anemic=1, else 0.
    """
    log.info("=== Processing Tongue dataset ===")
    records = []

    img_dir  = TONGUE_DIR / "dataset"
    mask_dir = TONGUE_DIR / "groundtruth" / "mask"

    if not img_dir.exists():
        img_dir = TONGUE_DIR
        log.warning("  'dataset/' not found — scanning India_Tongue/ directly.")

    if not TONGUE_DIR.exists():
        log.warning(f"  Tongue dir not found: {TONGUE_DIR}")
        return records

    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    log.info(f"  {len(image_files)} tongue images in '{img_dir.name}/'.")
    log.info(f"  Mask dir: {'found' if mask_dir.exists() else 'NOT found'}")

    for img_path in image_files:
        img = safe_imread(img_path)
        if img is None:
            continue
        img = apply_clahe(img)

        # ── Paired mask lookup ─────────────────────────────────────────────────
        mask = None
        if mask_dir.exists():
            for ext in (".png", ".bmp", ".jpg", ".jpeg", ".tif"):
                candidate = mask_dir / (img_path.stem + ext)
                if candidate.exists():
                    m = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        mask = (m > 127).astype(np.uint8) * 255
                        if mask.shape != img.shape[:2]:
                            mask = cv2.resize(
                                mask, (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                    break

        if mask is not None:
            feats = extract_color_features(img, mask)
        else:
            roi   = center_crop(img, ratio=0.55)
            feats = extract_color_features(roi)

        label = 1 if feats["lab_a"] < 145.0 else 0
        feats["label"]  = label
        feats["source"] = "tongue"
        records.append(feats)

    log.info(f"  Tongue: {len(records)} samples extracted.")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Nails Dataset
# ══════════════════════════════════════════════════════════════════════════════

NAIL_LABEL_MAP = {
    "healthy_nail":               0,
    "blue_finger":                1,
    "clubbing":                   1,
    "pitting":                    1,
    "acral_lentiginous_melanoma": 1,
    "onychogryphosis":            1,
}


def process_nails():
    """
    India_Nails/
        train/
            Acral_Lentiginous_Melanoma/ blue_finger/ clubbing/
            Healthy_Nail/ Onychogryphosis/ pitting/
        validation/  (same six subfolders)

    Label from class subfolder name. Random filenames are fine.
    ROI: center crop of the nail image.
    """
    log.info("=== Processing Nails dataset ===")
    records = []

    if not NAILS_DIR.exists():
        log.warning(f"  Nails dir not found: {NAILS_DIR}")
        return records

    split_dirs = [d for d in sorted(NAILS_DIR.iterdir()) if d.is_dir()]
    log.info(f"  Split folders: {[d.name for d in split_dirs]}")

    for split_dir in split_dirs:
        class_dirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
        if not class_dirs:
            log.warning(f"  No class subfolders in '{split_dir.name}/' — skipping.")
            continue

        for class_dir in class_dirs:
            folder_key = class_dir.name.lower()
            label      = NAIL_LABEL_MAP.get(folder_key, 1)

            img_files = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
            log.info(
                f"  [{split_dir.name}] {class_dir.name}: "
                f"{len(img_files)} images -> label={label}"
            )

            for img_path in img_files:
                img = safe_imread(img_path)
                if img is None:
                    continue
                img  = apply_clahe(img)
                roi  = center_crop(img, ratio=0.6)
                feats           = extract_color_features(roi)
                feats["label"]  = label
                feats["source"] = "nails"
                records.append(feats)

    log.info(f"  Nails: {len(records)} samples extracted.")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Tri-Check Data Engine starting ...")
    all_records = []

    all_records.extend(process_eyes())
    all_records.extend(process_tongue())
    all_records.extend(process_nails())

    if not all_records:
        log.error("No records extracted! Check that data folders contain images.")
        return

    df = pd.DataFrame(all_records)
    df = df[["source", "rg_ratio", "lab_a", "mean_r", "mean_g", "mean_b", "label"]]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(OUT_CSV, index=False)
    log.info(f"Saved {len(df)} rows -> {OUT_CSV}")
    log.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    log.info(f"Source distribution:\n{df['source'].value_counts().to_string()}")


if __name__ == "__main__":
    main()