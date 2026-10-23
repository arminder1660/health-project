"""CardiaNova - Clinical Entity Extraction

Rule-based extraction of clinical entities, symptoms, triggers, radiation
patterns and risk factors from the free-text descriptions patients type.

This module is deliberately a hand-written pattern matcher, not a trained
language model. Clinical keyword families are small, stable and safety
critical, so explicit, auditable rules are preferred over a statistical
model whose mistakes cannot be explained to a patient.
"""
from __future__ import annotations

import re
from typing import Any, Dict


def extract_clinical_entities(text: str) -> Dict[str, Any]:
    """Parses everyday human text into structured clinical signals.

    Designed to handle queries from young adults to elderly patients.
    """
    raw_lower = text.lower().strip()

    extracted: Dict[str, Any] = {
        "raw_text": text,
        "age": None,
        "gender": None,
        "systolic_bp": None,
        "diastolic_bp": None,
        "has_hypertension": False,
        "has_diabetes": False,
        "is_smoker": False,
        "has_family_history": False,
        "pain_character": [],
        "radiation": [],
        "triggers": [],
        "relief": [],
        "duration": "unknown",
        "red_flags": [],
        "associated_symptoms": [],
        "primary_symptom_category": "UNKNOWN",
    }

    # 1. Age extraction
    age_match = re.search(r'\b(?:i\s*am\s*|age\s*[:=]?\s*|aged\s*)?(\d{1,2})\s*(?:yo|y/o|years?\s*old|yr|yrs)?\b', raw_lower)
    if age_match:
        val = int(age_match.group(1))
        if 10 <= val <= 110:
            extracted["age"] = val

    # 2. Gender extraction
    if re.search(r'\b(female|woman|lady|girl|mother|wife|she|her)\b', raw_lower):
        extracted["gender"] = "FEMALE"
    elif re.search(r'\b(male|man|guy|boy|father|husband|he|him)\b', raw_lower):
        extracted["gender"] = "MALE"

    # 3. Blood Pressure extraction (e.g., "150/90", "140 / 85", "bp 160 100")
    bp_match = re.search(r'\b(?:bp|blood\s*pressure)?\s*[:=]?\s*(\d{2,3})\s*(?:/|\s+over\s+|\s+)\s*(\d{2,3})\b', raw_lower)
    if bp_match:
        sys = int(bp_match.group(1))
        dia = int(bp_match.group(2))
        if 70 <= sys <= 260 and 40 <= dia <= 160:
            extracted["systolic_bp"] = sys
            extracted["diastolic_bp"] = dia
            if sys >= 130 or dia >= 85:
                extracted["has_hypertension"] = True

    if re.search(r'\b(high\s*bp|hypertension|blood\s*pressure\s*tablets|high\s*pressure|hypertensive)\b', raw_lower):
        extracted["has_hypertension"] = True

    # 4. Diabetes / Sugar
    if re.search(r'\b(diabet(?:ic|es)|sugar|insulin|metformin|hba1c)\b', raw_lower):
        extracted["has_diabetes"] = True

    # 5. Smoking / Tobacco
    if re.search(r'\b(smoke|smoker|smoking|cigarettes?|vape|vaping|tobacco|bidi|nicotine)\b', raw_lower):
        if not re.search(r'\b(non-smoker|don\'t\s*smoke|never\s*smoke|quit\s*smoking)\b', raw_lower):
            extracted["is_smoker"] = True

    # 6. Family History
    if re.search(r'\b(family\s*history|father|mother|brother|sister|parents?)\s*(?:had|with)?\s*(?:(?:a|an|the)\s+)?(?:heart\s*attack|bypass|stent|cardiac|angina)\b', raw_lower):
        extracted["has_family_history"] = True

    # 7. Pain Character / Sensation
    if re.search(r'\b(heavy|heaviness|pressure|crushing|elephant|tightness|tight|squeezing|constriction|gripping)\b', raw_lower):
        extracted["pain_character"].append("HEAVY_PRESSURE_TIGHTNESS")

    if re.search(r'\b(burn|burning|heartburn|acid|sour|reflux|gerd|indigestion)\b', raw_lower):
        extracted["pain_character"].append("BURNING_ACID_REFLUX")

    if re.search(r'\b(sharp|stabbing|needle|prick|pleuritic|catches\s*when\s*breathing|hurts\s*to\s*press)\b', raw_lower):
        extracted["pain_character"].append("SHARP_PLEURITIC_OR_MUSCULOSKELETAL")

    if re.search(r'\b(flutter|fluttering|palpitation|palpitations|skipping|skipped\s*beats?|pounding|racing\s*heart|fast\s*pulse)\b', raw_lower):
        extracted["pain_character"].append("PALPITATIONS_ARRHYTHMIA")

    # 8. Pain Radiation
    if re.search(r'\b(left\s*arm|left\s*shoulder|left\s*hand)\b', raw_lower):
        extracted["radiation"].append("LEFT_ARM_SHOULDER")

    if re.search(r'\b(jaw|lower\s*teeth|neck|throat)\b', raw_lower):
        extracted["radiation"].append("JAW_NECK_THROAT")

    if re.search(r'\b(back|between\s*shoulder\s*blades?)\b', raw_lower):
        extracted["radiation"].append("INTERSCAPULAR_BACK")

    # 9. Triggers
    if re.search(r'\b(stairs|staircase|walking\s*(?:uphill|fast)?|climbing|exercise|exertion|lifting|effort|running)\b', raw_lower):
        extracted["triggers"].append("PHYSICAL_EXERTION")

    if re.search(r'\b(after\s*eating|heavy\s*meal|spicy|dinner|lunch|lying\s*down\s*after\s*meal)\b', raw_lower):
        extracted["triggers"].append("POST_PRANDIAL_MEALS")

    if re.search(r'\b(stress|anxiety|nervous|argument|panic|exam|work\s*pressure)\b', raw_lower):
        extracted["triggers"].append("EMOTIONAL_STRESS")

    if re.search(r'\b(coffee|tea|caffeine|energy\s*drink|red\s*bull)\b', raw_lower):
        extracted["triggers"].append("STIMULANTS_CAFFEINE")

    # 10. Relief Dynamics
    if re.search(r'\b(relieved\s*by\s*rest|stops\s*when\s*i\s*sit|better\s*when\s*resting|goes\s*away\s*with\s*rest|rest\s*helps)\b', raw_lower):
        extracted["relief"].append("RELIEVED_BY_REST")

    if re.search(r'\b(antacid|eno|gelusil|drinking\s*water|burping)\b', raw_lower):
        extracted["relief"].append("RELIEVED_BY_ANTACID")

    # 11. Duration
    if re.search(r'\b(few\s*seconds|momentary|split\s*second)\b', raw_lower):
        extracted["duration"] = "SECONDS"
    elif re.search(r'\b([2-9]|1[0-5])\s*min(?:ute)?s?\b', raw_lower):
        extracted["duration"] = "MINUTES_5_TO_15"
    elif re.search(r'\b(?:(?:2[0-9]|3[0-9]|4[0-9]|5[0-9])\s*min(?:ute)?s?|hours?|all\s*day|continuous|constant)\b', raw_lower):
        extracted["duration"] = "PROLONGED_OVER_20_MINS"

    # 12. Critical Red Flags (with negation handling)
    if re.search(r'\b(sweat|sweating|cold\s*sweat|perspiration|diaphoresis)\b', raw_lower):
        if not re.search(r'\b(no|not|without|zero)\s+(?:cold\s*)?(?:sweat|sweating|perspiration)\b', raw_lower):
            extracted["red_flags"].append("COLD_SWEATING_DIAPHORESIS")

    if re.search(r'\b(breathless|shortness\s*of\s*breath|gasping|can\'t\s*breathe|dyspnea)\b', raw_lower):
        if not re.search(r'\b(no|not|without)\s+(?:shortness\s*of\s*breath|breathlessness|gasping|dyspnea)\b', raw_lower):
            extracted["red_flags"].append("ACUTE_DYSPNEA")

    if re.search(r'\b(dizzy|dizziness|faint\w*|blackout|syncope|passed\s*out|lightheaded)\b', raw_lower):
        if not re.search(r'\b(no|not|without)\s+(?:dizziness|fainting|blackout|syncope|lightheadedness)\b', raw_lower):
            extracted["red_flags"].append("PRESYNCOPE_SYNCOPE")

    if re.search(r'\b(swollen\s*(?:feet|ankles|legs)|(?:feet|ankles|legs)\s*(?:are|is|look\w*|appear\w*|seem\w*)?\s*swollen|edema|water\s*in\s*legs)\b', raw_lower):
        if not re.search(r'\b(no|not|without)\s+(?:swollen|edema|swelling)\b', raw_lower):
            extracted["associated_symptoms"].append("PERIPHERAL_EDEMA")

    # Determine Primary Categorical Impression
    if "HEAVY_PRESSURE_TIGHTNESS" in extracted["pain_character"]:
        if "PHYSICAL_EXERTION" in extracted["triggers"] or "RELIEVED_BY_REST" in extracted["relief"]:
            extracted["primary_symptom_category"] = "SUSPECTED_STABLE_ANGINA"
        elif "PROLONGED_OVER_20_MINS" == extracted["duration"] or extracted["red_flags"]:
            extracted["primary_symptom_category"] = "POTENTIAL_ACUTE_CORONARY_SYNDROME"
        else:
            extracted["primary_symptom_category"] = "ISCHEMIC_HEART_CONCERN"
    elif "PALPITATIONS_ARRHYTHMIA" in extracted["pain_character"]:
        extracted["primary_symptom_category"] = "CARDIAC_ARRHYTHMIA_PALPITATIONS"
    elif "BURNING_ACID_REFLUX" in extracted["pain_character"]:
        extracted["primary_symptom_category"] = "GASTROESOPHAGEAL_REFLUX_GERD"
    elif "SHARP_PLEURITIC_OR_MUSCULOSKELETAL" in extracted["pain_character"]:
        extracted["primary_symptom_category"] = "MUSCULOSKELETAL_CHEST_WALL_STRAIN"
    elif extracted["has_hypertension"]:
        extracted["primary_symptom_category"] = "HYPERTENSIVE_VASCULAR_CONCERN"
    elif (extracted["pain_character"] or extracted["radiation"]
          or extracted["red_flags"] or extracted["associated_symptoms"]):
        extracted["primary_symptom_category"] = "NON_SPECIFIC_SYMPTOMS"
    # Otherwise no clinical signal was described at all, so the category stays
    # UNKNOWN rather than claiming a non-specific symptom was reported.

    return extracted
