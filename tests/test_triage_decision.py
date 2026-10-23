"""Tests for the clinical triage / specialist escalation engine.

These assertions encode the safety contract: the escalation level, the theme the
UI paints, and whether the patient is asked to consult a clinician.
"""
import pytest

from src.triage_decision import evaluate_triage_decision


def make_entities(**overrides):
    """A minimal well-formed entity dict, matching the extractor's output shape."""
    base = {
        "age": 45,
        "gender": "MALE",
        "systolic_bp": None,
        "diastolic_bp": None,
        "has_hypertension": False,
        "has_diabetes": False,
        "is_smoker": False,
        "pain_character": [],
        "radiation": [],
        "triggers": [],
        "relief": [],
        "duration": "unknown",
        "red_flags": [],
        "associated_symptoms": [],
        "primary_symptom_category": "UNKNOWN",
    }
    base.update(overrides)
    return base


def test_emergency_for_pressure_with_cold_sweats():
    entities = make_entities(
        pain_character=["HEAVY_PRESSURE_TIGHTNESS"],
        red_flags=["COLD_SWEATING_DIAPHORESIS"],
    )
    result = evaluate_triage_decision(entities)
    assert result["triage_level"] == "EMERGENCY_ER"
    assert result["theme"] == "blood-red"
    assert "IMMEDIATELY" in result["verdict_headline"]


def test_emergency_for_pressure_radiating_to_the_left_arm_when_prolonged():
    entities = make_entities(
        pain_character=["HEAVY_PRESSURE_TIGHTNESS"],
        radiation=["LEFT_ARM_SHOULDER"],
        duration="PROLONGED_OVER_20_MINS",
    )
    assert evaluate_triage_decision(entities)["triage_level"] == "EMERGENCY_ER"


def test_emergency_for_syncope_with_chest_pressure():
    entities = make_entities(
        pain_character=["HEAVY_PRESSURE_TIGHTNESS"],
        red_flags=["PRESYNCOPE_SYNCOPE"],
    )
    assert evaluate_triage_decision(entities)["triage_level"] == "EMERGENCY_ER"


def test_emergency_for_hypertensive_crisis():
    entities = make_entities(systolic_bp=185, diastolic_bp=125)
    result = evaluate_triage_decision(entities)
    assert result["triage_level"] == "EMERGENCY_ER"
    assert "Hypertensive Crisis" in result["primary_suspicion"]


def test_emergency_does_not_trigger_from_left_arm_alone():
    # Radiation without the pressure character and without a prolonged duration
    # must not be escalated to an emergency.
    entities = make_entities(radiation=["LEFT_ARM_SHOULDER"], duration="SECONDS")
    assert evaluate_triage_decision(entities)["triage_level"] != "EMERGENCY_ER"


def test_cardiologist_for_exertional_angina():
    entities = make_entities(
        pain_character=["HEAVY_PRESSURE_TIGHTNESS"],
        triggers=["PHYSICAL_EXERTION"],
    )
    result = evaluate_triage_decision(entities)
    assert result["triage_level"] == "SPECIALIST_CARDIOLOGIST"
    assert result["theme"] == "red"


def test_cardiologist_for_angina_relieved_by_rest():
    entities = make_entities(
        pain_character=["HEAVY_PRESSURE_TIGHTNESS"],
        relief=["RELIEVED_BY_REST"],
    )
    assert evaluate_triage_decision(entities)["triage_level"] == "SPECIALIST_CARDIOLOGIST"


def test_cardiologist_for_radiation_with_pressure():
    entities = make_entities(
        pain_character=["HEAVY_PRESSURE_TIGHTNESS"],
        radiation=["JAW_NECK_THROAT"],
    )
    assert evaluate_triage_decision(entities)["triage_level"] == "SPECIALIST_CARDIOLOGIST"


def test_cardiologist_for_palpitations_in_an_older_patient():
    entities = make_entities(pain_character=["PALPITATIONS_ARRHYTHMIA"], age=60)
    assert evaluate_triage_decision(entities)["triage_level"] == "SPECIALIST_CARDIOLOGIST"


def test_high_model_risk_escalates_only_with_a_risk_factor_present():
    entities = make_entities(
        pain_character=["HEAVY_PRESSURE_TIGHTNESS"],
        has_hypertension=True,
    )
    result = evaluate_triage_decision(entities, {"probability": 78.0})
    assert result["triage_level"] == "SPECIALIST_CARDIOLOGIST"


def test_high_model_risk_alone_does_not_escalate():
    # A high score with no symptom pattern and no risk factor must not
    # manufacture a cardiology referral.
    entities = make_entities(pain_character=["HEAVY_PRESSURE_TIGHTNESS"])
    result = evaluate_triage_decision(entities, {"probability": 91.0})
    assert result["triage_level"] != "SPECIALIST_CARDIOLOGIST"


def test_general_physician_for_hypertension_without_symptoms():
    entities = make_entities(has_hypertension=True)
    result = evaluate_triage_decision(entities)
    assert result["triage_level"] == "GENERAL_PHYSICIAN"
    assert result["theme"] == "yellow"


def test_general_physician_for_peripheral_edema():
    entities = make_entities(associated_symptoms=["PERIPHERAL_EDEMA"])
    assert evaluate_triage_decision(entities)["triage_level"] == "GENERAL_PHYSICIAN"


def test_home_care_for_musculoskeletal_pain():
    entities = make_entities(
        primary_symptom_category="MUSCULOSKELETAL_CHEST_WALL_STRAIN",
        pain_character=["SHARP_PLEURITIC_OR_MUSCULOSKELETAL"],
    )
    result = evaluate_triage_decision(entities)
    assert result["triage_level"] == "HOME_CARE_LIFESTYLE"
    assert result["theme"] == "green"


def test_every_level_returns_the_full_ui_contract():
    required = {
        "triage_level", "theme", "urgency_class", "risk_tier", "theme_label",
        "badge_color", "verdict_headline", "consult_big_doctor", "urgency",
        "primary_suspicion", "reasoning", "red_flag_warnings",
    }
    cases = [
        make_entities(pain_character=["HEAVY_PRESSURE_TIGHTNESS"], red_flags=["COLD_SWEATING_DIAPHORESIS"]),
        make_entities(pain_character=["HEAVY_PRESSURE_TIGHTNESS"], triggers=["PHYSICAL_EXERTION"]),
        make_entities(has_hypertension=True),
        make_entities(primary_symptom_category="MUSCULOSKELETAL_CHEST_WALL_STRAIN"),
    ]
    for entities in cases:
        result = evaluate_triage_decision(entities, {"probability": 30.0})
        assert required <= set(result.keys())
        assert result["red_flag_warnings"], "every level must tell the patient when to escalate"


def test_theme_matches_triage_level():
    # The UI paints itself from `theme`, so it must never disagree with the level.
    mapping = {
        "EMERGENCY_ER": "blood-red",
        "SPECIALIST_CARDIOLOGIST": "red",
        "GENERAL_PHYSICIAN": "yellow",
        "HOME_CARE_LIFESTYLE": "green",
    }
    cases = [
        make_entities(pain_character=["HEAVY_PRESSURE_TIGHTNESS"], red_flags=["COLD_SWEATING_DIAPHORESIS"]),
        make_entities(pain_character=["HEAVY_PRESSURE_TIGHTNESS"], triggers=["PHYSICAL_EXERTION"]),
        make_entities(has_hypertension=True),
        make_entities(primary_symptom_category="MUSCULOSKELETAL_CHEST_WALL_STRAIN"),
    ]
    for entities in cases:
        result = evaluate_triage_decision(entities, {"probability": 30.0})
        assert result["theme"] == mapping[result["triage_level"]]


@pytest.mark.parametrize("probability", [0.0, 50.0, 100.0])
def test_triage_never_crashes_on_any_model_score(probability):
    entities = make_entities(pain_character=["HEAVY_PRESSURE_TIGHTNESS"])
    assert evaluate_triage_decision(entities, {"probability": probability})["triage_level"]
