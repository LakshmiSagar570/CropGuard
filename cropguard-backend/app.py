from fastapi import FastAPI, File, UploadFile, Form, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import io, os, time
from datetime import datetime, timedelta, timezone
import re
import sqlite3
import json
import secrets
import hashlib
import hmac
import smtplib
from email.mime.text import MIMEText
from urllib.parse import quote
from urllib.request import Request, urlopen
import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

APP_START = time.time()

MODEL_PATH = os.environ.get("CROPGUARD_MODEL", "cropguard_model.h5")
CLASS_NAMES_PATH = os.environ.get("CROPGUARD_CLASS_NAMES", "class_names.txt")
DB_PATH = os.environ.get("CROPGUARD_DB", "diagnoses.db")
TOKEN_TTL_HOURS = int(os.environ.get("AUTH_TOKEN_TTL_HOURS", "168"))
OTP_TTL_MINUTES = int(os.environ.get("OTP_TTL_MINUTES", "10"))
OTP_DELIVERY = os.environ.get("OTP_DELIVERY", "mock").lower()
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER)

app = FastAPI(title="CropGuard AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = None
resolved_gemini_model = None
plant_gate_model = None
SUPPORTED_LANGUAGES = [
    "English",
    "Hindi",
    "Telugu",
    "Kannada",
    "Tamil",
    "Malayalam",
    "Marathi",
    "Gujarati",
    "Bengali",
    "Punjabi",
    "Odia",
    "Urdu",
]


def load_class_names(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    return names or None


def load_model():
    global model, class_names
    class_names = load_class_names(CLASS_NAMES_PATH)
    if TF_AVAILABLE and os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)


load_model()


def load_plant_gate_model():
    global plant_gate_model
    if not TF_AVAILABLE:
        return
    try:
        # Dedicated model for plant vs non-plant gating.
        plant_gate_model = tf.keras.applications.MobileNetV2(weights="imagenet")
    except Exception:
        plant_gate_model = None


load_plant_gate_model()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            crop TEXT NOT NULL,
            district TEXT NOT NULL,
            state TEXT NOT NULL,
            season TEXT NOT NULL,
            language TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            image_name TEXT,
            advice_excerpt TEXT
        )
        """
    )
    # Lightweight migration for user-scoped history
    cols = [r[1] for r in conn.execute("PRAGMA table_info(diagnoses)").fetchall()]
    if "user_phone" not in cols:
        conn.execute("ALTER TABLE diagnoses ADD COLUMN user_phone TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL UNIQUE,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            phone TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS password_reset_otps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            email TEXT,
            otp_hash TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            used INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    user_cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "email" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
    otp_cols = [r[1] for r in conn.execute("PRAGMA table_info(password_reset_otps)").fetchall()]
    if "email" not in otp_cols:
        conn.execute("ALTER TABLE password_reset_otps ADD COLUMN email TEXT")
    conn.commit()
    conn.close()


init_db()


def pick_confidence(preds):
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx] * 100.0)
    return top_idx, confidence


def normalize_crop_name(crop):
    c = (crop or "").strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c).strip("_")
    aliases = {
        "bell pepper": "pepper",
        "bell_pepper": "pepper",
        "pepper": "pepper",
        "potato": "potato",
        "tomato": "tomato",
        "rice": "rice",
        "paddy": "rice",
        "all": "all",
        "any": "all",
        "all_crops": "all",
    }
    return aliases.get(c, c)


def crop_token_from_class(class_name):
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (class_name or "")).strip("_").lower()
    if not cleaned:
        return ""
    return cleaned.split("_")[0]


def load_leaf_image(raw_bytes: bytes):
    src = Image.open(io.BytesIO(raw_bytes))
    # If image has alpha channel, crop to non-transparent content and composite on white.
    if src.mode in ("RGBA", "LA") or ("transparency" in src.info):
        rgba = src.convert("RGBA")
        alpha = rgba.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            rgba = rgba.crop(bbox)
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        src = Image.alpha_composite(bg, rgba).convert("RGB")
    else:
        src = src.convert("RGB")
    model_img = src.resize((224, 224))
    return src, model_img


def leaf_visual_metrics(rgb_img: Image.Image):
    arr = np.array(rgb_img).astype(np.uint8)
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)

    # Ignore near-white background pixels if present.
    non_bg = ~((r > 240) & (g > 240) & (b > 240))
    if non_bg.sum() == 0:
        return {"green_ratio": 0.0, "lesion_ratio": 1.0}

    # Green healthy tissue heuristic
    green_mask = non_bg & (g > r + 10) & (g > b + 10)
    green_ratio = float(green_mask.sum() / non_bg.sum())

    # Brown/black lesion heuristic
    brown_mask = non_bg & (r > 90) & (g > 40) & (g < r) & (b < g)
    dark_mask = non_bg & (r < 45) & (g < 45) & (b < 45)
    lesion_ratio = float((brown_mask | dark_mask).sum() / non_bg.sum())
    return {"green_ratio": green_ratio, "lesion_ratio": lesion_ratio}


def is_visually_healthy_leaf(metrics: dict):
    green = float(metrics.get("green_ratio", 0.0))
    lesion = float(metrics.get("lesion_ratio", 1.0))
    return green >= 0.92 and lesion <= 0.005


def indices_for_crop(crop):
    key = normalize_crop_name(crop)
    if not class_names:
        return []
    if key == "all":
        return list(range(len(class_names)))

    matched = []
    for i, name in enumerate(class_names):
        token = crop_token_from_class(name)
        if token == key:
            matched.append(i)
    return matched


def supported_crops():
    crop_set = []
    seen = set()
    for n in class_names or []:
        token = crop_token_from_class(n)
        if token and token not in seen:
            seen.add(token)
            crop_set.append(token.title())
    return crop_set


def infer_climate_context(state, district, season):
    season_key = (season or "").strip().lower()
    season_profile = {
        "kharif": "Monsoon-linked period in many Indian regions; humidity and leaf wetness are often high, increasing fungal and bacterial pressure.",
        "rabi": "Cooler and relatively drier period in many regions; dew and irrigation practices can still trigger foliar disease in dense canopies.",
        "zaid": "Hot and dry period between major seasons; heat stress, irrigation cycles, and intermittent humidity spikes can worsen leaf issues.",
    }
    regional_hint = (
        "In coastal and south-central India, humidity swings, warm temperatures, and irrigation timing can strongly affect leaf disease spread."
    )
    profile = season_profile.get(season_key, "Use current local weather patterns, humidity, and irrigation conditions to reason about disease pressure.")
    return (
        f"District: {district}\n"
        f"State: {state}\n"
        f"Current season: {season}\n"
        f"Date context: {datetime.now().strftime('%d %B %Y')}\n"
        f"Climate profile: {profile}\n"
        f"Regional hint: {regional_hint}"
    )


def normalize_language(language: str):
    raw = (language or "").strip().lower()
    mapping = {
        "english": "English",
        "hindi": "Hindi",
        "telugu": "Telugu",
        "kannada": "Kannada",
        "tamil": "Tamil",
        "malayalam": "Malayalam",
        "marathi": "Marathi",
        "gujarati": "Gujarati",
        "gujrati": "Gujarati",
        "bengali": "Bengali",
        "punjabi": "Punjabi",
        "odia": "Odia",
        "oriya": "Odia",
        "urdu": "Urdu",
    }
    return mapping.get(raw, "English")


def get_live_climate_context(state, district):
    try:
        query = quote(f"{district}, {state}, India")
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1&language=en&format=json"
        geo_req = Request(geo_url, headers={"User-Agent": "CropGuardAI/1.0"})
        with urlopen(geo_req, timeout=6) as resp:
            geo_data = json.loads(resp.read().decode("utf-8"))
        results = geo_data.get("results") or []
        if not results:
            return None
        lat = results[0]["latitude"]
        lon = results[0]["longitude"]

        w_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
            "&timezone=auto"
        )
        w_req = Request(w_url, headers={"User-Agent": "CropGuardAI/1.0"})
        with urlopen(w_req, timeout=6) as resp:
            w_data = json.loads(resp.read().decode("utf-8"))
        cur = w_data.get("current", {})
        return (
            f"Live weather now for {district}, {state}: "
            f"temperature {cur.get('temperature_2m', 'NA')}C, "
            f"humidity {cur.get('relative_humidity_2m', 'NA')}%, "
            f"precipitation {cur.get('precipitation', 'NA')} mm, "
            f"wind {cur.get('wind_speed_10m', 'NA')} km/h."
        )
    except Exception:
        return None


def build_prompt(crop, district, state, season, disease, confidence, language, climate_context):
    return (
        "You are an expert agricultural advisor for farmers in Andhra Pradesh, India.\n"
        "You receive a crop leaf image diagnosis from a CV model plus location context.\n"
        "Use both: (1) leaf disease signal and (2) location-climate reasoning.\n"
        "Give practical, simple advice that a farmer with basic education can follow.\n"
        "Always respond in the language specified. Be specific to the region and season.\n\n"
        f"Crop: {crop}\n"
        f"Disease from image model: {disease}\n"
        f"Confidence: {confidence:.0f}%\n"
        "Location and climate context:\n"
        f"{climate_context}\n"
        "Country: India\n"
        f"Language: {language}\n\n"
        "Write body text only in the specified language. Keep section headings in English exactly as below.\n"
        "Return ONLY these sections in this exact order with these exact headings:\n"
        "1. WHAT:\n"
        "2. WHY:\n"
        "3. NOW:\n"
        "4. PREVENT:\n"
        "5. URGENCY:\n"
        "Rules:\n"
        "- Keep each section concise and farmer-friendly.\n"
        "- NOW must be a clear 7-day action plan.\n"
        "- URGENCY must be exactly one of: Low, Medium, High.\n"
    )


def build_style_instruction(diagnosis_mode: str, response_detail: str):
    mode = (diagnosis_mode or "Balanced").strip().title()
    detail = (response_detail or "Standard").strip().title()
    mode_rule = "Use balanced caution."
    if mode == "Strict":
        mode_rule = "Be conservative. If confidence is not very high, emphasize uncertainty and verification."
    elif mode == "Fast":
        mode_rule = "Prioritize direct, practical action steps over long explanation."

    detail_rule = "Keep moderate detail."
    if detail == "Short":
        detail_rule = "Keep each section very short (1-2 concise sentences)."
    elif detail == "Detailed":
        detail_rule = "Give richer actionable details while staying practical."
    return f"{mode_rule} {detail_rule}"


def apply_detail_level(advice: str, response_detail: str):
    detail = (response_detail or "Standard").strip().title()
    if detail != "Short":
        return advice
    out = []
    for line in advice.splitlines():
        m = re.match(r"^(WHAT|WHY|NOW|PREVENT|URGENCY)\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        if not m:
            continue
        head = m.group(1).upper()
        body = m.group(2).strip()
        if head != "URGENCY":
            # keep only short leading chunk for concise mode
            body = body[:180].strip()
        out.append(f"{head}: {body}")
    return "\n".join(out) if out else advice


def build_uncertain_prompt(crop, district, state, season, language, climate_context, status):
    return (
        "You are an agricultural assistant for Indian farmers.\n"
        "The image diagnosis is uncertain or unsupported.\n"
        f"Crop: {crop}\n"
        f"District: {district}\n"
        f"State: {state}\n"
        f"Season: {season}\n"
        f"Status: {status}\n"
        f"Language: {language}\n"
        f"Climate context:\n{climate_context}\n\n"
        "Write body text only in the specified language. Keep section headings in English exactly as below.\n"
        "Return ONLY these sections in this exact order with these exact headings:\n"
        "1. WHAT:\n"
        "2. WHY:\n"
        "3. NOW:\n"
        "4. PREVENT:\n"
        "5. URGENCY:\n"
        "URGENCY must be exactly one of: Low, Medium, High.\n"
    )


def fallback_advice(language="English", uncertain=False):
    # Fallback is English-only; primary path should be Gemini in requested language.
    if uncertain:
        return (
            "WHAT: The diagnosis is uncertain from this image.\n"
            "WHY: The crop may be unsupported, the photo may be unclear, or symptoms are not distinct.\n"
            "NOW: Retake a clear close-up of a single affected leaf in daylight and submit again. If symptoms are spreading fast, consult a local agri officer today.\n"
            "PREVENT: Capture images early, avoid wet leaves at capture time, and monitor weekly for first signs.\n"
            "URGENCY: Medium"
        )
    return (
        "WHAT: This looks like a common leaf disease affecting crop health.\n"
        "WHY: Usually caused by humidity, overwatering, or fungal spread.\n"
        "NOW: Remove affected leaves, avoid overhead watering, apply recommended fungicide.\n"
        "PREVENT: Use disease-resistant varieties and improve airflow.\n"
        "URGENCY: Medium"
    )


def save_history(crop, district, state, season, language, disease, confidence, status, image_name, advice, user_phone):
    excerpt = (advice or "")[:500]
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO diagnoses
        (created_at, crop, district, state, season, language, disease, confidence, status, image_name, advice_excerpt, user_phone)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            crop,
            district,
            state,
            season,
            language,
            disease,
            float(confidence),
            status,
            image_name or "",
            excerpt,
            user_phone,
        ),
    )
    conn.commit()
    conn.close()


def now_utc():
    return datetime.now(timezone.utc)


def normalize_phone(phone: str):
    p = re.sub(r"[^\d+]", "", (phone or "").strip())
    if p.startswith("0"):
        p = p.lstrip("0")
    if not p.startswith("+"):
        p = f"+91{p}" if len(p) == 10 else f"+{p}"
    return p


def normalize_email(email: str):
    return (email or "").strip().lower()


def hash_password(password: str):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 120000).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str):
    try:
        salt, digest = stored.split("$", 1)
        check = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 120000).hex()
        return hmac.compare_digest(check, digest)
    except Exception:
        return False


def create_session(phone: str):
    token = secrets.token_urlsafe(40)
    created = now_utc()
    expires = created + timedelta(hours=TOKEN_TTL_HOURS)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions(token, phone, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (token, phone, expires.isoformat(), created.isoformat()),
    )
    conn.commit()
    conn.close()
    return token, expires


def get_current_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT token, phone, expires_at FROM sessions WHERE token = ?", (token,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid token")
    exp = datetime.fromisoformat(row["expires_at"])
    if exp < now_utc():
        raise HTTPException(status_code=401, detail="Session expired")
    return {"phone": row["phone"], "token": row["token"]}


def auth_required(authorization: str | None = Header(default=None)):
    return get_current_user(authorization)


def hash_otp(otp: str):
    return hashlib.sha256(otp.encode("utf-8")).hexdigest()


def send_otp(recipient_phone: str | None, recipient_email: str | None, otp: str):
    if OTP_DELIVERY == "email" and recipient_email:
        if not (SMTP_HOST and SMTP_USER and SMTP_PASS and SMTP_FROM):
            return {"sent": False, "provider": "email", "error": "SMTP not configured"}
        subject = "CropGuard OTP for Password Reset"
        body = (
            "Your CropGuard password reset OTP is:\n\n"
            f"{otp}\n\n"
            f"This OTP expires in {OTP_TTL_MINUTES} minutes.\n"
            "If you did not request this, ignore this email."
        )
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg["To"] = recipient_email
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.sendmail(SMTP_FROM, [recipient_email], msg.as_string())
            return {"sent": True, "provider": "email"}
        except Exception as e:
            return {"sent": False, "provider": "email", "error": str(e)}
    # Fallback mock for hackathon/dev.
    print(f"[OTP] phone={recipient_phone or 'NA'} email={recipient_email or 'NA'} otp={otp}")
    return {"sent": True, "provider": "mock"}


class RegisterInput(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    phone: str = Field(min_length=8, max_length=20)
    email: str = Field(min_length=5, max_length=120)
    password: str = Field(min_length=6, max_length=64)


class LoginInput(BaseModel):
    phone: str = Field(min_length=8, max_length=20)
    password: str = Field(min_length=6, max_length=64)


class ForgotPasswordRequestInput(BaseModel):
    email: str = Field(min_length=5, max_length=120)


class ForgotPasswordVerifyInput(BaseModel):
    email: str = Field(min_length=5, max_length=120)
    otp: str = Field(min_length=4, max_length=8)
    new_password: str = Field(min_length=6, max_length=64)


def detect_non_plant_with_gate_model(pil_img):
    if plant_gate_model is None or not TF_AVAILABLE:
        return {"available": False, "is_non_plant": False, "reason": "gate_model_unavailable"}
    try:
        img = pil_img.resize((224, 224))
        arr = np.array(img).astype(np.float32)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(arr, axis=0))
        preds = plant_gate_model.predict(arr, verbose=0)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]
        top_label = decoded[0][1].lower()
        top_conf = float(decoded[0][2] * 100.0)
        plant_tokens = [
            "leaf", "tree", "flower", "plant", "corn", "maize", "wheat", "rice", "fungus",
            "mushroom", "fruit", "vegetable", "herb", "tomato", "potato", "pepper",
            "grape", "apple", "strawberry", "cherry", "peach", "orange", "banana",
            "cabbage", "broccoli", "cauliflower", "sunflower", "daisy", "rose", "lily",
            "petal", "blossom", "vine", "grass",
        ]
        has_plant_signal = any(any(tok in label.lower() for tok in plant_tokens) for _, label, _ in decoded)
        # Keep this conservative: only strong non-plant predictions should block diagnosis.
        is_non_plant = (not has_plant_signal) and top_conf >= 75.0
        return {
            "available": True,
            "is_non_plant": is_non_plant,
            "top_label": top_label,
            "top_confidence": top_conf,
        }
    except Exception:
        return {"available": False, "is_non_plant": False, "reason": "gate_model_failed"}


def normalize_structured_advice(text, language="English", uncertain=False):
    if not text:
        return fallback_advice(language=language, uncertain=uncertain)

    lines = text.splitlines()
    sections = {"WHAT": [], "WHY": [], "NOW": [], "PREVENT": [], "URGENCY": []}
    current = None
    heading_re = re.compile(r"^\s*(?:\d+\s*[\.\)]\s*)?(WHAT|WHY|NOW|PREVENT|URGENCY)\s*:\s*(.*)$", re.IGNORECASE)

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = heading_re.match(line)
        if m:
            current = m.group(1).upper()
            tail = m.group(2).strip()
            if tail:
                sections[current].append(tail)
        elif current:
            sections[current].append(line)

    if not all(sections[k] for k in sections):
        return fallback_advice(language=language, uncertain=uncertain)

    # Keep URGENCY to one line for easier UI display.
    urgency = sections["URGENCY"][0]
    return (
        f"WHAT: {' '.join(sections['WHAT'])}\n"
        f"WHY: {' '.join(sections['WHY'])}\n"
        f"NOW: {' '.join(sections['NOW'])}\n"
        f"PREVENT: {' '.join(sections['PREVENT'])}\n"
        f"URGENCY: {urgency}"
    )


def set_urgency_in_advice(advice: str, urgency: str):
    return re.sub(r"(?im)^URGENCY\s*:.*$", f"URGENCY: {urgency}", advice)


def apply_urgency_guardrails(advice: str, disease: str, status: str, confidence: float):
    d = (disease or "").lower()
    # Healthy classes should never be high urgency.
    if "healthy" in d:
        return set_urgency_in_advice(advice, "Low")
    # Uncertain/unsupported/non-plant should not claim high urgency.
    if status in {"uncertain", "unsupported_crop", "possible_non_plant"}:
        return set_urgency_in_advice(advice, "Medium")
    # Mid-confidence disease predictions should be capped at Medium.
    if confidence < 85:
        return set_urgency_in_advice(advice, "Medium")
    return advice


def pick_label_with_healthy_guard(class_names_list, crop_indices, filtered_probs, diagnosis_mode: str):
    order = np.argsort(filtered_probs)[::-1]
    top_local = int(order[0])
    top_idx = crop_indices[top_local]
    top_conf = float(filtered_probs[top_local] * 100.0)
    top_name = class_names_list[top_idx] if class_names_list and top_idx < len(class_names_list) else f"Class_{top_idx}"

    # Find healthy class for the selected crop, if available.
    healthy_idx = None
    healthy_conf = 0.0
    for local_i, global_idx in enumerate(crop_indices):
        name = class_names_list[global_idx].lower() if class_names_list and global_idx < len(class_names_list) else ""
        if "healthy" in name:
            healthy_idx = global_idx
            healthy_conf = float(filtered_probs[local_i] * 100.0)
            break

    mode = (diagnosis_mode or "Balanced").strip().title()
    # Guardrail to reduce false disease alarms on healthy leaves.
    if healthy_idx is not None and "healthy" not in top_name.lower():
        margin = top_conf - healthy_conf
        if mode == "Strict":
            if healthy_conf >= 22.0 and margin <= 15.0:
                return healthy_idx, healthy_conf
        elif mode == "Balanced":
            if healthy_conf >= 28.0 and margin <= 10.0:
                return healthy_idx, healthy_conf
        else:  # Fast mode
            if healthy_conf >= 35.0 and margin <= 8.0:
                return healthy_idx, healthy_conf

    return top_idx, top_conf


def resolve_gemini_model():
    global resolved_gemini_model
    if resolved_gemini_model:
        return resolved_gemini_model

    env_model = os.environ.get("GEMINI_MODEL")
    if env_model:
        resolved_gemini_model = env_model
        return resolved_gemini_model

    candidates = []
    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            model_name = getattr(m, "name", "")
            if model_name.startswith("models/"):
                model_name = model_name.split("/", 1)[1]
            candidates.append(model_name)

    if not candidates:
        resolved_gemini_model = "gemini-1.5-flash"
        return resolved_gemini_model

    for c in candidates:
        if "flash" in c.lower():
            resolved_gemini_model = c
            return resolved_gemini_model

    resolved_gemini_model = candidates[0]
    return resolved_gemini_model


def generate_advice(prompt, language="English", uncertain=False):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not GENAI_AVAILABLE:
        return fallback_advice(language=language, uncertain=uncertain)
    genai.configure(api_key=api_key)
    try:
        model_name = resolve_gemini_model()
        llm = genai.GenerativeModel(model_name)
        response = llm.generate_content(prompt)
        raw = getattr(response, "text", None)
        return normalize_structured_advice(raw, language=language, uncertain=uncertain)
    except Exception:
        return fallback_advice(language=language, uncertain=uncertain)


@app.post("/auth/register")
async def auth_register(payload: RegisterInput):
    phone = normalize_phone(payload.phone)
    email = normalize_email(payload.email)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    existing = conn.execute("SELECT id FROM users WHERE phone = ? OR email = ?", (phone, email)).fetchone()
    if existing:
        conn.close()
        return JSONResponse(status_code=409, content={"error": "Phone or email already registered"})
    conn.execute(
        "INSERT INTO users(name, phone, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
        (payload.name.strip(), phone, email, hash_password(payload.password), now_utc().isoformat()),
    )
    conn.commit()
    conn.close()
    token, expires = create_session(phone)
    return {
        "token": token,
        "expires_at": expires.isoformat(),
        "user": {"name": payload.name.strip(), "phone": phone, "email": email},
    }


@app.post("/auth/login")
async def auth_login(payload: LoginInput):
    phone = normalize_phone(payload.phone)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT name, phone, email, password_hash FROM users WHERE phone = ?", (phone,)).fetchone()
    conn.close()
    if not row or not verify_password(payload.password, row["password_hash"]):
        return JSONResponse(status_code=401, content={"error": "Invalid phone or password"})
    token, expires = create_session(phone)
    return {
        "token": token,
        "expires_at": expires.isoformat(),
        "user": {"name": row["name"], "phone": row["phone"], "email": row["email"]},
    }


@app.get("/auth/me")
async def auth_me(user=Depends(auth_required)):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT name, phone, email, created_at FROM users WHERE phone = ?", (user["phone"],)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": dict(row)}


@app.post("/auth/logout")
async def auth_logout(user=Depends(auth_required)):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM sessions WHERE token = ?", (user["token"],))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/auth/forgot-password/request")
async def forgot_password_request(payload: ForgotPasswordRequestInput):
    email = normalize_email(payload.email)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    exists = conn.execute("SELECT id, phone, email FROM users WHERE email = ?", (email,)).fetchone()
    if not exists:
        conn.close()
        return {"ok": True}
    otp = f"{secrets.randbelow(1000000):06d}"
    expires = now_utc() + timedelta(minutes=OTP_TTL_MINUTES)
    conn.execute("UPDATE password_reset_otps SET used = 1 WHERE email = ? AND used = 0", (email,))
    conn.execute(
        "INSERT INTO password_reset_otps(phone, email, otp_hash, expires_at, attempts, used, created_at) VALUES (?, ?, ?, ?, 0, 0, ?)",
        (exists["phone"], email, hash_otp(otp), expires.isoformat(), now_utc().isoformat()),
    )
    conn.commit()
    conn.close()
    send_result = send_otp(exists["phone"], email, otp)
    if not send_result.get("sent"):
        return JSONResponse(status_code=500, content={"error": f"OTP delivery failed: {send_result.get('error', 'unknown')}"})
    return {"ok": True, "message": f"OTP sent via {send_result.get('provider', 'mock')}"}


@app.post("/auth/forgot-password/verify")
async def forgot_password_verify(payload: ForgotPasswordVerifyInput):
    email = normalize_email(payload.email)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT id, otp_hash, expires_at, attempts, used
        FROM password_reset_otps
        WHERE email = ?
        ORDER BY id DESC LIMIT 1
        """,
        (email,),
    ).fetchone()
    if not row:
        conn.close()
        return JSONResponse(status_code=400, content={"error": "No OTP request found"})
    if row["used"] == 1:
        conn.close()
        return JSONResponse(status_code=400, content={"error": "OTP already used"})
    if datetime.fromisoformat(row["expires_at"]) < now_utc():
        conn.close()
        return JSONResponse(status_code=400, content={"error": "OTP expired"})
    if row["attempts"] >= 5:
        conn.close()
        return JSONResponse(status_code=429, content={"error": "Too many attempts"})
    if not hmac.compare_digest(hash_otp(payload.otp), row["otp_hash"]):
        conn.execute("UPDATE password_reset_otps SET attempts = attempts + 1 WHERE id = ?", (row["id"],))
        conn.commit()
        conn.close()
        return JSONResponse(status_code=400, content={"error": "Invalid OTP"})
    conn.execute("UPDATE users SET password_hash = ? WHERE email = ?", (hash_password(payload.new_password), email))
    phone_row = conn.execute("SELECT phone FROM users WHERE email = ?", (email,)).fetchone()
    if phone_row:
        conn.execute("DELETE FROM sessions WHERE phone = ?", (phone_row["phone"],))
    conn.execute("UPDATE password_reset_otps SET used = 1 WHERE id = ?", (row["id"],))
    conn.commit()
    conn.close()
    return {"ok": True, "message": "Password updated"}


@app.get("/health")
async def health():
    return {"status": "ok", "uptime_sec": int(time.time() - APP_START)}


@app.get("/supported-crops")
async def get_supported_crops():
    return {"supported_crops": supported_crops(), "class_count": len(class_names or [])}


@app.get("/history")
async def get_history(limit: int = 20, user=Depends(auth_required)):
    safe_limit = max(1, min(limit, 100))
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, created_at, crop, district, state, season, language, disease, confidence, status
        FROM diagnoses
        WHERE user_phone = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user["phone"], safe_limit),
    ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows]}


@app.post("/diagnose")
async def diagnose(
    file: UploadFile = File(...),
    crop: str = Form(...),
    district: str = Form(...),
    state: str = Form("Andhra Pradesh"),
    season: str = Form("Kharif"),
    language: str = Form("English"),
    diagnosis_mode: str = Form("Balanced"),
    response_detail: str = Form("Standard"),
    user=Depends(auth_required),
):
    try:
        language = normalize_language(language)
        raw = await file.read()
        orig_img, model_img = load_leaf_image(raw)
        metrics = leaf_visual_metrics(orig_img)
        img_array = np.expand_dims(np.array(model_img) / 255.0, axis=0)
        plant_gate = detect_non_plant_with_gate_model(model_img)
        live_weather = get_live_climate_context(state, district)
        climate_context = infer_climate_context(state, district, season)
        if live_weather:
            climate_context = f"{climate_context}\n{live_weather}"

        if model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Model not loaded. Place cropguard_model.h5 and class_names.txt in backend folder.",
                    "tf_available": TF_AVAILABLE,
                },
            )

        preds = model.predict(img_array)[0]
        overall_conf = float(np.max(preds) * 100.0)
        crop_indices = indices_for_crop(crop)
        top_predictions = []
        status = "ok"
        visual_override = False

        if not crop_indices:
            disease = "Uncertain"
            confidence = 0.0
            status = "unsupported_crop"
        else:
            filtered = preds[crop_indices]
            order = np.argsort(filtered)[::-1]
            for j in order[:2]:
                idx = crop_indices[int(j)]
                top_predictions.append(
                    {
                        "disease": class_names[idx] if class_names and idx < len(class_names) else f"Class_{idx}",
                        "confidence": float(filtered[int(j)] * 100.0),
                    }
                )
            top_idx, confidence = pick_label_with_healthy_guard(class_names, crop_indices, filtered, diagnosis_mode)
            disease = class_names[top_idx] if class_names and top_idx < len(class_names) else f"Class_{top_idx}"

            # Visual sanity override: if leaf looks very clean green with negligible lesions,
            # force healthy class for this crop to avoid over-calling blight on pristine leaves.
            healthy_local = None
            for li, gi in enumerate(crop_indices):
                n = class_names[gi].lower() if class_names and gi < len(class_names) else ""
                if "healthy" in n:
                    healthy_local = li
                    break
            if (
                healthy_local is not None
                and "healthy" not in disease.lower()
                and is_visually_healthy_leaf(metrics)
            ):
                healthy_idx = crop_indices[healthy_local]
                healthy_conf = float(filtered[healthy_local] * 100.0)
                disease = class_names[healthy_idx]
                # If model underestimates healthy on pristine leaf, keep a practical confidence floor.
                confidence = max(healthy_conf, 75.0)
                visual_override = True

            if confidence < 65:
                disease = "Uncertain"
                status = "uncertain"
            elif confidence <= 85 and len(top_predictions) > 1:
                status = "needs_confirmation"

        # Dedicated model gate first; heuristic fallback second.
        if plant_gate.get("available") and plant_gate.get("is_non_plant") and overall_conf < 55.0:
            disease = "Uncertain"
            confidence = overall_conf
            status = "possible_non_plant"
        elif overall_conf < 40 and status == "ok":
            disease = "Uncertain"
            confidence = overall_conf
            status = "possible_non_plant"

        if status in {"uncertain", "unsupported_crop", "possible_non_plant"}:
            uncertain_prompt = build_uncertain_prompt(
                crop=crop,
                district=district,
                state=state,
                season=season,
                language=language,
                climate_context=climate_context,
                status=status,
            )
            uncertain_prompt = f"{uncertain_prompt}\nStyle: {build_style_instruction(diagnosis_mode, response_detail)}"
            advice = generate_advice(uncertain_prompt, language=language, uncertain=True)
        else:
            prompt = build_prompt(crop, district, state, season, disease, confidence, language, climate_context)
            prompt = f"{prompt}\nStyle: {build_style_instruction(diagnosis_mode, response_detail)}"
            advice = generate_advice(prompt, language=language, uncertain=False)
        advice = apply_detail_level(advice, response_detail)
        advice = apply_urgency_guardrails(advice, disease, status, confidence)

        save_history(
            crop=crop,
            district=district,
            state=state,
            season=season,
            language=language,
            disease=disease,
            confidence=confidence,
            status=status,
            image_name=getattr(file, "filename", ""),
            advice=advice,
            user_phone=user["phone"],
        )

        return {
            "disease": disease,
            "confidence": confidence,
            "overall_confidence": overall_conf,
            "advice": advice,
            "language": language,
            "diagnosis_mode": diagnosis_mode,
            "response_detail": response_detail,
            "state": state,
            "district": district,
            "status": status,
            "top_predictions": top_predictions,
            "supported_crops": supported_crops(),
            "climate_context": climate_context,
            "plant_gate": plant_gate,
            "image_metrics": metrics,
            "visual_override": visual_override,
            "model_loaded": True,
            "tf_available": TF_AVAILABLE,
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
