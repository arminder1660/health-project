"""CardiaNova - Clinical Triage & Specialist Escalation Engine

Generates an evidence-based clinical opinion on whether the patient must
consult an emergency hospital, a specialist cardiologist, a primary care physician,
or can manage their condition with home monitoring and diet.
"""
from typing import Any, Dict


def evaluate_triage_decision(entities: Dict[str, Any], ml_risk: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Evaluates extracted clinical entities and outputs a clear triage verdict.
    
    Levels:
    - EMERGENCY_ER (🚨)
    - SPECIALIST_CARDIOLOGIST (⚠️ - "Consult a Big Doctor")
    - GENERAL_PHYSICIAN (🟡)
    - HOME_CARE_LIFESTYLE (🟢)
    """
    category = entities.get("primary_symptom_category", "UNKNOWN")
    pain_char = entities.get("pain_character", [])
    radiation = entities.get("radiation", [])
    triggers = entities.get("triggers", [])
    relief = entities.get("relief", [])
    red_flags = entities.get("red_flags", [])
    duration = entities.get("duration", "unknown")
    age = entities.get("age") or 45
    sys_bp = entities.get("systolic_bp") or (120 if not entities.get("has_hypertension") else 145)
    dia_bp = entities.get("diastolic_bp") or (80 if not entities.get("has_hypertension") else 92)
    has_htn = entities.get("has_hypertension", False)
    is_smoker = entities.get("is_smoker", False)
    has_diabetes = entities.get("has_diabetes", False)

    # 1. Check for Immediate Emergency (Acute Myocardial Infarction / Crisis)
    is_emergency = False
    emergency_reasons = []

    if "COLD_SWEATING_DIAPHORESIS" in red_flags and "HEAVY_PRESSURE_TIGHTNESS" in pain_char:
        is_emergency = True
        emergency_reasons.append("Heavy chest tightness accompanied by unexplained cold drenching sweats.")

    if "LEFT_ARM_SHOULDER" in radiation and "HEAVY_PRESSURE_TIGHTNESS" in pain_char and duration == "PROLONGED_OVER_20_MINS":
        is_emergency = True
        emergency_reasons.append("Prolonged retrosternal crushing pain radiating directly to the left arm/shoulder.")

    if "PRESYNCOPE_SYNCOPE" in red_flags and "HEAVY_PRESSURE_TIGHTNESS" in pain_char:
        is_emergency = True
        emergency_reasons.append("Chest discomfort combined with fainting, syncope, or severe dizziness.")

    if sys_bp >= 180 or dia_bp >= 120:
        is_emergency = True
        emergency_reasons.append(f"Severely elevated blood pressure ({sys_bp}/{dia_bp} mm Hg) matching Hypertensive Crisis.")

    if is_emergency:
        return {
            "triage_level": "EMERGENCY_ER",
            "theme": "blood-red",
            "urgency_class": "blood-red",
            "risk_tier": "extreme",
            "theme_label": "High Extreme Risk",
            "badge_color": "#7f1d1d",  # Blood Red
            "verdict_headline": "CRITICAL EMERGENCY — GO TO THE NEAREST HOSPITAL ER IMMEDIATELY",
            "consult_big_doctor": "CRITICAL EMERGENCY: Immediate emergency hospital care required. Call 911 / 112 / 108 or have someone drive you to the nearest ER immediately. Do NOT delay.",
            "urgency": "IMMEDIATE EMERGENCY (0–60 Minutes)",
            "primary_suspicion": "Possible Acute Coronary Syndrome (Heart Attack) or Hypertensive Crisis",
            "reasoning": (
                "Your reported symptoms include cardinal signs of acute myocardial ischemia or dangerous cardiovascular compromise. "
                "The combination of " + "; ".join(emergency_reasons) + " requires immediate emergency medical evaluation, 12-lead ECG, and cardiac troponin blood testing."
            ),
            "emergency_actions": [
                "Call your local emergency number (911 / 112 / 108) or have someone drive you to the nearest hospital ER.",
                "Do NOT drive yourself under any circumstance.",
                "Sit upright in a comfortable, propped-up recovery position to ease breathing and heart workload.",
                "Loosen tight clothing around your neck and chest.",
                "If recommended by an emergency dispatcher and you have no allergy, chew one adult aspirin (325 mg) or 4 baby aspirins (81 mg each).",
            ],
            "red_flag_warnings": [
                "Sudden loss of consciousness or speech difficulty",
                "Severe shortness of breath making speaking impossible",
                "Worsening crushing sensation spreading to jaw or both arms",
            ]
        }

    # 2. Check for "Big Doctor" Specialist (Cardiologist) Recommendation
    is_cardiologist = False
    cardio_reasons = []

    # Exertional angina pattern
    if "HEAVY_PRESSURE_TIGHTNESS" in pain_char and ("PHYSICAL_EXERTION" in triggers or "RELIEVED_BY_REST" in relief):
        is_cardiologist = True
        cardio_reasons.append("Chest pressure provoked by walking, climbing stairs, or effort and relieved by rest (Classic Stable Angina).")

    # Radiation to arm or jaw without acute prolonged emergency
    if ("LEFT_ARM_SHOULDER" in radiation or "JAW_NECK_THROAT" in radiation) and "HEAVY_PRESSURE_TIGHTNESS" in pain_char:
        is_cardiologist = True
        cardio_reasons.append("Pain radiating to anatomical cardiac dermatomes (left arm, neck, or lower jaw).")

    # Arrhythmia / Palpitations with dizziness or older age
    if "PALPITATIONS_ARRHYTHMIA" in pain_char and (age >= 45 or "PRESYNCOPE_SYNCOPE" in red_flags or has_htn):
        is_cardiologist = True
        cardio_reasons.append("Heart flutterings or skipped beats with associated dizziness or underlying vascular risk.")

    # High cumulative cardiovascular risk profile (when ischemic or vascular indicators present)
    if ml_risk and ml_risk.get("probability", 0) >= 60.0 and (has_htn or is_smoker) and ("HEAVY_PRESSURE_TIGHTNESS" in pain_char or category in ("HYPERTENSIVE_VASCULAR_CONCERN", "ISCHEMIC_HEART_CONCERN")):
        is_cardiologist = True
        cardio_reasons.append(f"Calibrated risk model projects an elevated {ml_risk.get('probability')}% baseline cardiovascular disease probability.")

    if is_cardiologist:
        return {
            "triage_level": "SPECIALIST_CARDIOLOGIST",
            "theme": "red",
            "urgency_class": "red",
            "risk_tier": "risk",
            "theme_label": "In Risk (Cardiologist Needed)",
            "badge_color": "#dc2626",  # Alert Red
            "verdict_headline": "IN RISK — CONSULT A SPECIALIST CARDIOLOGIST WITHIN 48 TO 72 HOURS",
            "consult_big_doctor": "YES. In risk: You should see an experienced Cardiologist (Heart Specialist) rather than relying solely on home remedies.",
            "urgency": "URGENT OUTPATIENT (Within 2–3 Days)",
            "primary_suspicion": "Suspected Coronary Artery Ischemia (Angina Pectoris) or Cardiac Rhythm Disturbance",
            "reasoning": (
                "Your reported pattern includes specific cardiovascular indicators: " + "; ".join(cardio_reasons) + " "
                "Because these symptoms point to possible reduced blood flow through the coronary arteries under exertion, "
                "a specialist cardiology evaluation is clinically warranted to prevent a future acute cardiac event."
            ),
            "specialist_recommendations": [
                "Book an outpatient appointment with an Interventional or General Cardiologist.",
                "Expected Tests: Resting 12-lead ECG, 2D Transthoracic Echocardiogram (Echo), and a Treadmill Stress Test (TMT).",
                "Avoid strenuous exertion, competitive sports, or heavy lifting until cleared by your doctor.",
                "Keep a written log of when tightness occurs, what you were doing, and how many minutes it took to subside.",
            ],
            "red_flag_warnings": [
                "If the tightness suddenly becomes severe, lasts longer than 15 minutes while resting, or causes cold sweats, proceed to the Emergency Room immediately.",
            ]
        }

    # 3. Check for Primary Care / General Physician
    is_primary_care = False
    pc_reasons = []

    if has_htn or sys_bp >= 135 or dia_bp >= 88:
        is_primary_care = True
        pc_reasons.append(f"Blood pressure in the hypertensive range ({sys_bp}/{dia_bp} mm Hg).")

    if "BURNING_ACID_REFLUX" in pain_char and (age >= 50 or has_htn):
        is_primary_care = True
        pc_reasons.append("Upper chest burning occurring with vascular risk factors (needs confirmation that it is purely gastrointestinal).")

    if "PERIPHERAL_EDEMA" in entities.get("associated_symptoms", []):
        is_primary_care = True
        pc_reasons.append("Swelling in lower extremities (ankles/legs) requiring renal and vascular review.")

    if is_primary_care or category == "HYPERTENSIVE_VASCULAR_CONCERN":
        return {
            "triage_level": "GENERAL_PHYSICIAN",
            "theme": "yellow",
            "urgency_class": "yellow",
            "risk_tier": "moderate",
            "theme_label": "Moderate Risk",
            "badge_color": "#ca8a04",  # Warm Golden Yellow
            "verdict_headline": "MODERATE RISK — START WITH A GENERAL PHYSICIAN WITHIN 1 TO 2 WEEKS",
            "consult_big_doctor": "NOT IMMEDIATELY. Moderate risk detected: Start with a thorough checkup by a General Physician / Primary Care clinic.",
            "urgency": "ROUTINE REVIEW (Within 7–14 Days)",
            "primary_suspicion": "Hypertension Management, Metabolic Screening, or Gastrointestinal Reflux",
            "reasoning": (
                "Your symptoms do not show immediate signs of acute arterial blockage, but there are notable baseline health factors: "
                + "; ".join(pc_reasons) + " "
                "A general physician can review your vitals, order standard fasting blood work, and recommend blood pressure or lifestyle therapy."
            ),
            "clinic_recommendations": [
                "Visit a local primary care clinic or family physician within 1 to 2 weeks.",
                "Take 2 home blood pressure readings daily (morning and evening) and bring the 7-day log to your doctor.",
                "Ask for a standard lipid panel (HDL, LDL, Triglycerides) and fasting blood glucose / HbA1c test.",
                "Adopt the cardioprotective DASH diet outlined below.",
            ],
            "red_flag_warnings": [
                "If you develop chest tightness that worsens when walking or radiates to your left arm, escalate to a cardiologist.",
            ]
        }

    # 4. Home Care & Lifestyle Monitoring (Benign / Non-Cardiac - No Doctor Needed)
    return {
        "triage_level": "HOME_CARE_LIFESTYLE",
        "theme": "green",
        "urgency_class": "green",
        "risk_tier": "no-doctor",
        "theme_label": "Safe (No Doctor Needed)",
        "badge_color": "#059669",  # Emerald Green
        "verdict_headline": "SAFE FOR HOME CARE & LIFESTYLE MONITORING (NO DOCTOR NEEDED RIGHT NOW)",
        "consult_big_doctor": "NO. You do not need to consult a doctor at this time. Your symptoms are consistent with non-cardiac, benign causes.",
        "urgency": "SELF-CARE & LIFESTYLE MONITORING",
        "primary_suspicion": (
            "Musculoskeletal Chest Wall Soreness (Costochondritis), Acid Reflux, or Benign Stress-Induced Flutters"
            if category != "UNKNOWN" else "Mild Non-Specific Fatigue or Strain"
        ),
        "reasoning": (
            "Your description lacks ischemic hallmarks (no exertional chest pressure, no radiation to arm/jaw, no breathlessness or cold sweats). "
            "Sharp pricking pains that hurt when pressing the ribs, or brief flutters linked to caffeine/stress, are generally benign and respond well to rest, hydration, and diet adjustments."
        ),
        "home_care_recommendations": [
            "Rest and avoid straining or lifting heavy objects if your chest wall muscles are sore.",
            "Reduce caffeine, energy drinks, and late-night spicy meals.",
            "Stay well hydrated with 2 to 2.5 liters of water daily.",
            "Practice diaphragmatic deep breathing (5 seconds in, 5 seconds out) for 10 minutes to ease stress-related tension.",
        ],
        "red_flag_warnings": [
            "If your symptoms change to a heavy central crushing ache, spread to your throat or arm, or persist continuously, seek clinical review.",
        ]
    }
