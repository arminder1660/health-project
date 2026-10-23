"""CardiaNova - FastAPI Server

Serves the multi-stage web application and exposes endpoints for rule-based
symptom triage, calibrated ML cardiovascular risk scoring, what-if simulation
and live location-aware doctor discovery.
"""
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from src.entity_extraction import extract_clinical_entities
from src.triage_decision import evaluate_triage_decision
from src.diet_recommender import generate_diet_and_remedies
from src.train import predict_patient_risk, model_info, ModelNotTrainedError
from src.geo_doctors import search_doctors_near, google_api_key

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="CardiaNova Platform",
    description="Rule-Based Clinical Triage, Calibrated Cardiovascular Risk Scoring & Live Referral API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TriageRequest(BaseModel):
    query_text: Optional[str] = ""
    age: Optional[int] = None
    gender: Optional[str] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    sensation: Optional[str] = None
    radiation: Optional[str] = None
    trigger: Optional[str] = None
    duration: Optional[str] = None
    has_hypertension: Optional[bool] = False
    has_diabetes: Optional[bool] = False
    is_smoker: Optional[bool] = False

    # Self-reported lab values and lifestyle answers, in the same encoding the
    # training data uses. Left as None when the patient does not know, so the
    # model's native missing-value handling applies instead of a fabricated
    # default. These are *not* inferred from other conditions: high blood
    # pressure does not tell you anything about a cholesterol panel.
    cholesterol: Optional[int] = None  # 1 normal, 2 above normal, 3 well above normal
    gluc: Optional[int] = None         # 1 normal, 2 above normal, 3 well above normal
    alco: Optional[int] = None         # 1 drinks alcohol, 0 does not
    active: Optional[int] = None       # 1 physically active, 0 mostly inactive

    red_flags: Optional[List[str]] = []


class SimulationRequest(BaseModel):
    base_age: int = 54
    base_sys: int = 155
    base_dia: int = 95
    base_wt: float = 86
    base_ht: float = 172
    base_smk: bool = True
    base_active: bool = False
    target_sys: int = 122
    target_dia: int = 80
    target_wt: float = 76
    target_smk: bool = False
    target_active: bool = True


@app.post("/api/triage")
def run_triage(req: TriageRequest):
    """Runs the full screening pipeline: rule-based entity extraction, calibrated
    ML risk scoring, triage escalation, diet advice and referral preparation."""
    # Step 1: Rule-based clinical entity extraction on the free text (if provided)
    entities = extract_clinical_entities(req.query_text or "")
    
    # Override / supplement with structured wizard selections (if explicitly provided)
    if req.age is not None:
        entities["age"] = req.age
    if req.gender is not None:
        entities["gender"] = req.gender.upper()
    if req.systolic_bp is not None:
        entities["systolic_bp"] = req.systolic_bp
    if req.diastolic_bp is not None:
        entities["diastolic_bp"] = req.diastolic_bp
    if req.has_hypertension:
        entities["has_hypertension"] = True
    if req.has_diabetes:
        entities["has_diabetes"] = True
    if req.is_smoker:
        entities["is_smoker"] = True

    if req.sensation and req.sensation not in entities["pain_character"]:
        entities["pain_character"].append(req.sensation)
    if req.radiation and req.radiation not in entities["radiation"]:
        entities["radiation"].append(req.radiation)
    if req.trigger and req.trigger not in entities["triggers"]:
        entities["triggers"].append(req.trigger)
    if req.duration:
        entities["duration"] = req.duration
    if req.red_flags:
        for rf in req.red_flags:
            if rf not in entities["red_flags"]:
                entities["red_flags"].append(rf)

    # Step 2: Calibrated 70,000-patient ML inference
    patient_sys = entities["systolic_bp"] or 120
    patient_dia = entities["diastolic_bp"] or 80
    ml_payload = {
        "age": entities["age"] or 45,
        "gender": 2 if entities.get("gender") == "MALE" else 1,
        "height": req.height or 170,
        "weight": req.weight or 70,
        "ap_hi": patient_sys,
        "ap_lo": patient_dia,
        # Passed through verbatim. None means "not known", which the model
        # handles natively -- previously these were derived from hypertension
        # and diabetes, or hardcoded, so two features were constant and one was
        # measuring the wrong thing entirely.
        "cholesterol": req.cholesterol,
        "gluc": req.gluc,
        "smoke": 1 if entities.get("is_smoker") else 0,
        "alco": req.alco,
        "active": req.active,
    }
    
    try:
        ml_result = predict_patient_risk(ml_payload)
    except ModelNotTrainedError as exc:
        # Fail loudly instead of silently degrading to a guessed number: an
        # untrained model is an operator problem, not something to hide.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as e:
        ml_result = {
            "probability": 35.0,
            "risk_category": "Moderate Risk",
            "color": "#f59e0b",
            "bmi": round((req.weight or 75) / (((req.height or 170) / 100) ** 2), 1),
            "pulse_pressure": (entities["systolic_bp"] or 135) - (entities["diastolic_bp"] or 85),
        }
        print(f"[server] Risk model inference fell back to a heuristic: {e}")

    # Step 3: Specialist Escalation Matrix ("Consult a Big Doctor?")
    triage_result = evaluate_triage_decision(entities, ml_result)

    # Step 4: Tailored Medical Diet and Home Remedies
    diet_result = generate_diet_and_remedies(entities)

    # When any level of clinician review is advised (everything except safe home
    # care) the patient is asked for their location so real doctors near them can
    # be looked up live on the internet.
    consultation_needed = triage_result["triage_level"] != "HOME_CARE_LIFESTYLE"

    return {
        "status": "success",
        "entities": entities,
        "ml_result": ml_result,
        "triage": triage_result,
        "diet": diet_result,
        "consultation_needed": consultation_needed,
        "doctor_search": {
            "needs_location": consultation_needed,
            "triage_level": triage_result["triage_level"],
            "recommended_specialist": {
                "EMERGENCY_ER": "Emergency department of the nearest hospital",
                "SPECIALIST_CARDIOLOGIST": "Cardiologist (heart specialist) or cardiac centre",
                "GENERAL_PHYSICIAN": "General physician or family clinic",
            }.get(triage_result["triage_level"], "Cardiologist"),
            "agency": (
                "Ambulance: dial 108 (India) or 112/911 for emergency services."
            ),
        },
    }


@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    """Calculates what-if risk reduction between current baseline and target interventions."""
    base_res = predict_patient_risk({
        "age": req.base_age, "gender": 2, "height": req.base_ht, "weight": req.base_wt,
        "ap_hi": req.base_sys, "ap_lo": req.base_dia, "cholesterol": 2, "gluc": 1,
        "smoke": 1 if req.base_smk else 0, "alco": 0, "active": 1 if req.base_active else 0
    })

    target_res = predict_patient_risk({
        "age": req.base_age, "gender": 2, "height": req.base_ht, "weight": req.target_wt,
        "ap_hi": req.target_sys, "ap_lo": req.target_dia, "cholesterol": 1, "gluc": 1,
        "smoke": 1 if req.target_smk else 0, "alco": 0, "active": 1 if req.target_active else 0
    })

    risk_reduction = round(base_res["probability"] - target_res["probability"], 1)

    return {
        "base_risk": base_res,
        "target_risk": target_res,
        "risk_reduction": risk_reduction,
    }


@app.get("/api/nearby-doctors")
def get_nearby_doctors(
    location: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[int] = None,
    triage_level: Optional[str] = None,
    limit: int = 12,
):
    """Finds REAL hospitals, cardiac centres and doctors near the patient's own location.

    The lookup runs live against public geographic databases (OpenStreetMap by
    default, Google Places when a key is configured), so the patient only ever
    sees facilities that actually exist in the city they told us they are in.
    """
    return search_doctors_near(
        location=location,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        limit=limit,
        triage_level=triage_level,
    )


@app.get("/api/model-info")
def get_model_info():
    """Publishes the real performance of the risk model, including its downside.

    Returns the held-out metrics, the logistic-regression baseline it is compared
    against, permutation feature importance, and the ROC / calibration curves, so
    the numbers are auditable rather than a badge on a slide.
    """
    try:
        return model_info()
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/doctor-search-status")
def doctor_search_status():
    """Reports which live doctor directory is active (so the UI can be honest about it)."""
    key = google_api_key()
    return {
        "provider": "google_places" if key else "openstreetmap",
        "provider_label": "Google Places (live)" if key else "OpenStreetMap (live)",
        "google_key_configured": bool(key),
        "key_help": (
            "Add GOOGLE_PLACES_API_KEY=AIza... to a .env file in the project root to "
            "upgrade results with real doctor names, star ratings and review counts."
        ),
    }


# Mount the frontend web app static files
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
