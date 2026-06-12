"""
Tri-Check Anemia Triage — Flask Application
=============================================
Handles three image uploads (Eye, Tongue, Nail), extracts color features
using OpenCV (CLAHE + Haar cascades + center crop), runs the trained DNN,
and returns a Triage Assessment: Healthy | Moderate Risk | Severe Risk.

Run: py -3.11 app.py
"""

import os
import cv2
import base64
import logging
import numpy as np
import joblib
from pathlib import Path
from functools import lru_cache

from flask import Flask, render_template, request, jsonify

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── App & Paths ────────────────────────────────────────────────────────────────
app         = Flask(__name__)
BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "models" / "deep_anemia_model.h5"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── OpenCV Haar cascades ───────────────────────────────────────────────────────
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ── Triage thresholds ─────────────────────────────────────────────────────────
THRESHOLD_SEVERE   = 0.65
THRESHOLD_MODERATE = 0.40


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run  py -3.11 src/train_model.py  first."
        )
    log.info(f"Loading model from {MODEL_PATH} ...")
    return tf.keras.models.load_model(str(MODEL_PATH))


@lru_cache(maxsize=1)
def load_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. "
            "Run  py -3.11 src/train_model.py  first."
        )
    return joblib.load(SCALER_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# Image utilities
# ══════════════════════════════════════════════════════════════════════════════

def bytes_to_bgr(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def apply_clahe(bgr_img):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def extract_color_features(bgr_img, mask=None):
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
        "rg_ratio": float(rg_ratio),
        "lab_a":    float(lab_a),
        "mean_r":   float(mean_r),
        "mean_g":   float(mean_g),
        "mean_b":   float(mean_b),
    }


def center_crop(img, ratio=0.5):
    h, w   = img.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y1     = (h - ch) // 2
    x1     = (w - cw) // 2
    return img[y1:y1 + ch, x1:x1 + cw]


def bottom_center_crop(img, h_ratio=0.4, w_ratio=0.6):
    h, w   = img.shape[:2]
    ch, cw = int(h * h_ratio), int(w * w_ratio)
    y1     = h - ch
    x1     = (w - cw) // 2
    return img[y1:y1 + ch, x1:x1 + cw]


def bgr_to_base64(bgr_img):
    _, buf = cv2.imencode(".png", bgr_img)
    return base64.b64encode(buf).decode("utf-8")


def draw_box(img, x, y, w, h, color=(0, 255, 100)):
    vis = img.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    return vis


# ══════════════════════════════════════════════════════════════════════════════
# Per-channel processors
# ══════════════════════════════════════════════════════════════════════════════

def process_eye_image(file_bytes):
    img = bytes_to_bgr(file_bytes)
    if img is None:
        return {}, "Image decode error", ""
    img  = apply_clahe(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis  = img.copy()

    # Try Haar eye detection
    eyes   = _EYE_CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=4, minSize=(20, 20))
    status = "No eye detected — using bottom-center crop"
    roi    = bottom_center_crop(img)

    if len(eyes) > 0:
        eyes   = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)
        x, y, w, h = eyes[0]
        # Lower portion of eye box = conjunctiva
        y_c  = y + int(h * 0.55)
        h_c  = int(h * 0.45)
        roi  = img[y_c:y_c + h_c, x:x + w]
        if roi.size == 0:
            roi = bottom_center_crop(img)
        vis    = draw_box(img, x, y_c, w, h_c, color=(0, 255, 100))
        status = f"Eye detected — conjunctiva ROI ({x},{y_c})"

    feats = extract_color_features(roi)
    b64   = bgr_to_base64(vis)
    return feats, status, b64


def process_tongue_image(file_bytes):
    img = bytes_to_bgr(file_bytes)
    if img is None:
        return {}, "Image decode error", ""
    img    = apply_clahe(img)
    roi    = center_crop(img, ratio=0.55)
    status = "Center crop applied for tongue ROI"

    # Draw crop box on vis
    h, w   = img.shape[:2]
    ch, cw = int(h * 0.55), int(w * 0.55)
    y1     = (h - ch) // 2
    x1     = (w - cw) // 2
    vis    = draw_box(img, x1, y1, cw, ch, color=(255, 80, 80))

    feats = extract_color_features(roi)
    b64   = bgr_to_base64(vis)
    return feats, status, b64


def process_nail_image(file_bytes):
    img    = bytes_to_bgr(file_bytes)
    if img is None:
        return {}, "Image decode error", ""
    img    = apply_clahe(img)
    roi    = center_crop(img, ratio=0.6)
    status = "Center crop applied for nail ROI"

    h, w   = img.shape[:2]
    ch, cw = int(h * 0.6), int(w * 0.6)
    y1     = (h - ch) // 2
    x1     = (w - cw) // 2
    vis    = draw_box(img, x1, y1, cw, ch, color=(140, 80, 255))

    feats = extract_color_features(roi)
    b64   = bgr_to_base64(vis)
    return feats, status, b64


# ══════════════════════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_ORDER = ["rg_ratio", "lab_a", "mean_r", "mean_g", "mean_b"]


def predict_channel(feats):
    model  = load_model()
    scaler = load_scaler()
    row    = np.array([[feats[k] for k in FEATURE_ORDER]], dtype=np.float32)
    row_sc = scaler.transform(row)
    return float(model.predict(row_sc, verbose=0)[0][0])


def triage_label(score):
    if score >= THRESHOLD_SEVERE:
        return (
            "Severe Risk", "danger",
            "High likelihood of anemia. Please consult a medical professional immediately.",
        )
    elif score >= THRESHOLD_MODERATE:
        return (
            "Moderate Risk", "warning",
            "Some indicators of anemia present. Consider follow-up blood tests.",
        )
    else:
        return (
            "Healthy", "success",
            "No significant anemia indicators detected. Maintain a balanced diet.",
        )


def generate_recommendations(eye_prob, tongue_prob, nail_prob, combined_score):
    """
    Generate smart per-channel and overall recommendations based on probabilities.
    All probs are 0-100 scale.
    """
    recs = []

    # ── Overall severity recommendations ──────────────────────────────────────
    if combined_score >= 80:
        recs.append({
            "icon": "🚨",
            "level": "danger",
            "title": "Visit a Doctor Immediately",
            "text": (
                "Your combined anemia risk score is critically high ("
                + str(round(combined_score, 1))
                + "%). Please visit a hospital or clinic immediately for a "
                "complete blood count (CBC) test. Do not delay."
            ),
        })
    elif combined_score >= 65:
        recs.append({
            "icon": "⚠️",
            "level": "warning",
            "title": "See a Doctor Soon",
            "text": (
                "Your risk score ("
                + str(round(combined_score, 1))
                + "%) indicates significant anemia risk. Schedule a blood "
                "test with your doctor within the next few days."
            ),
        })
    elif combined_score >= 40:
        recs.append({
            "icon": "🩺",
            "level": "warning",
            "title": "Monitor and Follow Up",
            "text": (
                "Moderate indicators detected ("
                + str(round(combined_score, 1))
                + "%). Consider getting a haemoglobin test done at a "
                "nearby lab as a precaution."
            ),
        })
    else:
        recs.append({
            "icon": "✅",
            "level": "success",
            "title": "Looking Healthy",
            "text": (
                "No significant anemia indicators detected ("
                + str(round(combined_score, 1))
                + "%). Keep maintaining a balanced, iron-rich diet."
            ),
        })

    # ── Eye-specific recommendations ───────────────────────────────────────────
    if eye_prob >= 70:
        recs.append({
            "icon": "👁️",
            "level": "danger",
            "title": "Pale Conjunctiva Detected",
            "text": (
                "Your inner eyelid appears significantly pale ("
                + str(eye_prob)
                + "% risk). This is a strong clinical sign of low haemoglobin. "
                "Pull down your lower eyelid — if it looks white or very light "
                "pink instead of deep pink/red, consult a doctor urgently."
            ),
        })
    elif eye_prob >= 40:
        recs.append({
            "icon": "👁️",
            "level": "warning",
            "title": "Mild Pallor in Eyes",
            "text": (
                "Some pallor detected in the conjunctiva ("
                + str(eye_prob)
                + "% risk). Increase iron-rich foods (spinach, lentils, red "
                "meat) and vitamin C to help iron absorption."
            ),
        })

    # ── Tongue-specific recommendations ───────────────────────────────────────
    if tongue_prob >= 70:
        recs.append({
            "icon": "👅",
            "level": "danger",
            "title": "Pale / Smooth Tongue Detected",
            "text": (
                "Your tongue shows significant pallor ("
                + str(tongue_prob)
                + "% risk). A pale, smooth tongue losing its texture (papillae) "
                "is a classic sign of iron or B12 deficiency anaemia. "
                "Get a blood test including B12 and folate levels."
            ),
        })
    elif tongue_prob >= 40:
        recs.append({
            "icon": "👅",
            "level": "warning",
            "title": "Tongue Colour Slightly Off",
            "text": (
                "Mild tongue pallor detected ("
                + str(tongue_prob)
                + "% risk). Consider adding vitamin B12 sources (eggs, dairy, "
                "fish) and folate (leafy greens) to your diet."
            ),
        })

    # ── Nail-specific recommendations ─────────────────────────────────────────
    if nail_prob >= 70:
        recs.append({
            "icon": "💅",
            "level": "danger",
            "title": "Nail Abnormality Detected",
            "text": (
                "Your nail shows high risk indicators ("
                + str(nail_prob)
                + "% risk). This could indicate iron deficiency, poor circulation, "
                "or nail disease. First, ensure your nails are clean and free "
                "of nail polish before retesting. If pallor persists, see a doctor."
            ),
        })
    elif nail_prob >= 40:
        recs.append({
            "icon": "💅",
            "level": "warning",
            "title": "Check Your Nails",
            "text": (
                "Mild nail risk detected ("
                + str(nail_prob)
                + "% risk). Make sure your nails are clean and unpolished "
                "for accurate results. Press your nail firmly for 2 seconds "
                "and release — it should turn pink quickly. Slow return "
                "to pink may indicate poor circulation."
            ),
        })
    elif nail_prob < 40 and eye_prob >= 50:
        recs.append({
            "icon": "💅",
            "level": "info",
            "title": "Nails Look Fine",
            "text": (
                "Your nail bed appears healthy ("
                + str(nail_prob)
                + "% risk). However, other indicators are elevated — "
                "keep nails clean and recheck if symptoms persist."
            ),
        })

    # ── Isolated nail issue (nail high, others low) ────────────────────────────
    if nail_prob >= 50 and eye_prob < 35 and tongue_prob < 35:
        recs.append({
            "icon": "🧼",
            "level": "info",
            "title": "Likely a Nail Hygiene Issue",
            "text": (
                "Your eye and tongue appear healthy but the nail score is "
                "elevated ("
                + str(nail_prob)
                + "%). This pattern often means the nail image was affected "
                "by dirt, nail polish, or poor lighting rather than actual "
                "anaemia. Clean your nails thoroughly, remove any polish, "
                "and retake the photo in good natural light."
            ),
        })

    # ── Diet recommendations (always shown if any risk) ───────────────────────
    if combined_score >= 40:
        recs.append({
            "icon": "🥗",
            "level": "info",
            "title": "Dietary Recommendations",
            "text": (
                "Increase iron-rich foods: spinach, lentils, kidney beans, "
                "tofu, red meat, pumpkin seeds. Pair with vitamin C sources "
                "(lemon, orange, amla) to boost iron absorption. Avoid tea "
                "or coffee immediately after meals as they block iron uptake."
            ),
        })

    return recs


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    required = ("eye_image", "tongue_image", "nail_image")
    for field in required:
        if field not in request.files or request.files[field].filename == "":
            return render_template(
                "index.html",
                error=f"Missing upload: {field.replace('_', ' ').title()}",
            )

    channels = {
        "eye":    (request.files["eye_image"],    process_eye_image),
        "tongue": (request.files["tongue_image"], process_tongue_image),
        "nail":   (request.files["nail_image"],   process_nail_image),
    }

    results = {}
    probs   = []

    try:
        for ch_name, (file_obj, processor) in channels.items():
            ext = Path(file_obj.filename).suffix.lower()
            if ext not in ALLOWED_EXT:
                return render_template(
                    "index.html",
                    error=f"Unsupported format for {ch_name}: {ext}",
                )
            file_bytes          = file_obj.read()
            feats, status, b64  = processor(file_bytes)

            if not feats:
                return render_template(
                    "index.html", error=f"Could not process {ch_name} image."
                )

            prob   = predict_channel(feats)
            probs.append(prob)

            results[ch_name] = {
                "features":  {k: round(feats[k], 4) for k in FEATURE_ORDER},
                "prob":      round(prob * 100, 1),
                "status":    status,
                "image_b64": b64,
            }

        combined_score        = float(np.mean(probs))
        label, badge, message = triage_label(combined_score)

        eye_prob    = results["eye"]["prob"]
        tongue_prob = results["tongue"]["prob"]
        nail_prob   = results["nail"]["prob"]
        score_pct   = round(combined_score * 100, 1)

        recommendations = generate_recommendations(
            eye_prob, tongue_prob, nail_prob, score_pct
        )

        return render_template(
            "result.html",
            results=results,
            combined_score=score_pct,
            label=label,
            badge=badge,
            message=message,
            eye_prob=eye_prob,
            tongue_prob=tongue_prob,
            nail_prob=nail_prob,
            recommendations=recommendations,
        )

    except FileNotFoundError as exc:
        log.error(str(exc))
        return render_template("index.html", error=str(exc))
    except Exception as exc:
        log.exception("Prediction error")
        return render_template("index.html", error=f"Internal error: {exc}")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_ready": MODEL_PATH.exists()})


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("Starting Tri-Check Anemia Triage server ...")
    log.info(f"  Model  : {MODEL_PATH}")
    log.info(f"  Scaler : {SCALER_PATH}")
    log.info("  Open   : http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
