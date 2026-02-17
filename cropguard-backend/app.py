from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io, os, time
from datetime import datetime
import re
import sqlite3
import json
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


def fallback_advice():
    return (
        "WHAT: This looks like a common leaf disease affecting crop health.\n"
        "WHY: Usually caused by humidity, overwatering, or fungal spread.\n"
        "NOW: Remove affected leaves, avoid overhead watering, apply recommended fungicide.\n"
        "PREVENT: Use disease-resistant varieties and improve airflow.\n"
        "URGENCY: Medium"
    )


def fallback_uncertain_advice():
    return (
        "WHAT: The diagnosis is uncertain from this image.\n"
        "WHY: The crop may be unsupported, the photo may be unclear, or symptoms are not distinct.\n"
        "NOW: Retake a clear close-up of a single affected leaf in daylight and submit again. If symptoms are spreading fast, consult a local agri officer today.\n"
        "PREVENT: Capture images early, avoid wet leaves at capture time, and monitor weekly for first signs.\n"
        "URGENCY: Medium"
    )


def save_history(crop, district, state, season, language, disease, confidence, status, image_name, advice):
    excerpt = (advice or "")[:500]
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO diagnoses
        (created_at, crop, district, state, season, language, disease, confidence, status, image_name, advice_excerpt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ),
    )
    conn.commit()
    conn.close()


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


def normalize_structured_advice(text):
    if not text:
        return fallback_advice()

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
        return fallback_advice()

    # Keep URGENCY to one line for easier UI display.
    urgency = sections["URGENCY"][0]
    return (
        f"WHAT: {' '.join(sections['WHAT'])}\n"
        f"WHY: {' '.join(sections['WHY'])}\n"
        f"NOW: {' '.join(sections['NOW'])}\n"
        f"PREVENT: {' '.join(sections['PREVENT'])}\n"
        f"URGENCY: {urgency}"
    )


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


def generate_advice(prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not GENAI_AVAILABLE:
        return fallback_advice()
    genai.configure(api_key=api_key)
    try:
        model_name = resolve_gemini_model()
        llm = genai.GenerativeModel(model_name)
        response = llm.generate_content(prompt)
        raw = getattr(response, "text", None)
        return normalize_structured_advice(raw)
    except Exception:
        return fallback_advice()


@app.get("/health")
async def health():
    return {"status": "ok", "uptime_sec": int(time.time() - APP_START)}


@app.get("/supported-crops")
async def get_supported_crops():
    return {"supported_crops": supported_crops(), "class_count": len(class_names or [])}


@app.get("/history")
async def get_history(limit: int = 20):
    safe_limit = max(1, min(limit, 100))
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, created_at, crop, district, state, season, language, disease, confidence, status
        FROM diagnoses
        ORDER BY id DESC
        LIMIT ?
        """,
        (safe_limit,),
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
):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        plant_gate = detect_non_plant_with_gate_model(img)
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
            top_idx = crop_indices[int(order[0])]
            confidence = float(filtered[int(order[0])] * 100.0)
            disease = class_names[top_idx] if class_names and top_idx < len(class_names) else f"Class_{top_idx}"
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
            advice = fallback_uncertain_advice()
        else:
            prompt = build_prompt(crop, district, state, season, disease, confidence, language, climate_context)
            advice = generate_advice(prompt)

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
        )

        return {
            "disease": disease,
            "confidence": confidence,
            "overall_confidence": overall_conf,
            "advice": advice,
            "language": language,
            "state": state,
            "district": district,
            "status": status,
            "top_predictions": top_predictions,
            "supported_crops": supported_crops(),
            "climate_context": climate_context,
            "plant_gate": plant_gate,
            "model_loaded": True,
            "tf_available": TF_AVAILABLE,
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
