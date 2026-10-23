"""End-to-end tests for the FastAPI surface.

Tests that need the trained model artifact are skipped when `models/cardio_model.joblib`
is absent, so the suite stays runnable on a fresh clone. CI trains the model first,
which is why these run for real there.
"""
import pytest
from fastapi.testclient import TestClient

from src.train import MODEL_SAVE_PATH

import server

client = TestClient(server.app)

requires_model = pytest.mark.skipif(
    not MODEL_SAVE_PATH.exists(),
    reason="run `python -m src.train` to enable the model-dependent tests",
)

EMERGENCY_TEXT = "heavy crushing chest pressure with cold sweating for 30 minutes"
LOW_RISK_TEXT = (
    "sharp needle pricking pain in my chest when I press it, "
    "no sweating, no breathlessness, no dizziness"
)


# --------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------

def test_home_page_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "CardiaNova" in response.text


def test_removed_legacy_route_is_gone():
    # The fabricated /api/doctors endpoint was deleted on purpose.
    assert client.get("/api/doctors").status_code == 404


def test_health_of_the_docs_endpoint():
    assert client.get("/docs").status_code == 200


# --------------------------------------------------------------------------
# Doctor search status
# --------------------------------------------------------------------------

def test_doctor_search_status_reports_the_active_provider():
    payload = client.get("/api/doctor-search-status").json()
    assert payload["provider"] in ("openstreetmap", "google_places")
    assert isinstance(payload["google_key_configured"], bool)
    assert payload["provider_label"]


def test_nearby_doctors_asks_for_a_location_when_none_is_given():
    payload = client.get("/api/nearby-doctors").json()
    assert payload["status"] == "error"
    assert payload["doctors"] == []


def test_nearby_doctors_rejects_an_unresolvable_place(monkeypatch):
    from src import geo_doctors

    monkeypatch.setattr(geo_doctors, "geocode_place", lambda place: None)
    payload = client.get("/api/nearby-doctors", params={"location": "Nowhereville-xyz-999"}).json()
    assert payload["status"] == "error"
    assert payload["doctors"] == []


# --------------------------------------------------------------------------
# Model card
# --------------------------------------------------------------------------

@requires_model
def test_model_info_publishes_metrics_and_diagnostics():
    payload = client.get("/api/model-info").json()

    metrics = payload["metrics"]
    assert 0 < metrics["accuracy"] <= 100
    assert 0 < metrics["roc_auc"] <= 100
    assert 0 < metrics["brier_score"] < 1

    # A metric without a baseline is not evidence, so both must be present.
    assert payload["baseline_metrics"]["roc_auc"] > 0

    # The diagnostics the UI draws must actually be populated.
    assert len(payload["feature_importance"]) == len(payload["feature_names"])
    assert len(payload["roc_curve"]) > 5
    assert len(payload["calibration_curve"]) > 5

    # The model card must state its own limits.
    assert payload["caveats"]


@requires_model
def test_feature_importance_is_ordered_and_labelled():
    payload = client.get("/api/model-info").json()
    importances = [f["importance"] for f in payload["feature_importance"]]
    assert importances == sorted(importances, reverse=True)
    assert all(f["label"] for f in payload["feature_importance"])
    assert {"x", "y"} <= set(payload["roc_curve"][0].keys())


@requires_model
def test_pulse_pressure_is_not_a_model_feature():
    """Pulse pressure is a deterministic function of two features already in the
    set and measured at +0.0001 ROC-AUC importance, so it was dropped."""
    payload = client.get("/api/model-info").json()
    assert "pulse_pressure" not in payload["feature_names"]


@requires_model
def test_model_info_publishes_cross_validation_with_confidence_intervals():
    payload = client.get("/api/model-info").json()
    cv = payload["cross_validation"]

    assert cv["folds"] == 25  # 5 folds x 5 repeats
    for metric in ("accuracy", "roc_auc", "brier_score"):
        stats = cv[metric]
        assert stats["ci_low"] < stats["mean"] < stats["ci_high"]
        assert stats["std"] >= 0


@requires_model
def test_model_info_publishes_operating_point_metrics():
    payload = client.get("/api/model-info").json()
    op = payload["operating_point"]

    assert 0 < op["threshold"] < 1
    for key in ("sensitivity", "specificity", "ppv", "npv", "prevalence"):
        assert 0 < op[key] <= 100

    # Every held-out patient must land in exactly one cell.
    assert sum(op["confusion"].values()) == payload["metrics"]["test_samples"]


@requires_model
def test_confusion_counts_agree_with_the_reported_sensitivity():
    payload = client.get("/api/model-info").json()
    op = payload["operating_point"]
    confusion = op["confusion"]

    positives = confusion["true_positive"] + confusion["false_negative"]
    expected = confusion["true_positive"] / positives * 100
    assert abs(expected - op["sensitivity"]) < 0.01


@requires_model
def test_subgroup_metrics_cover_age_and_sex():
    payload = client.get("/api/model-info").json()
    subgroups = payload["subgroup_metrics"]

    assert {"Age band", "Sex"} == {g["dimension"] for g in subgroups}
    assert all(g["n"] > 0 for g in subgroups)

    # The sex subgroups must partition the held-out set exactly.
    male_or_female = sum(g["n"] for g in subgroups if g["dimension"] == "Sex")
    assert male_or_female == payload["metrics"]["test_samples"]


# --------------------------------------------------------------------------
# Triage
# --------------------------------------------------------------------------

@requires_model
def test_emergency_text_escalates_and_asks_for_a_location():
    payload = client.post("/api/triage", json={"query_text": EMERGENCY_TEXT}).json()

    assert payload["status"] == "success"
    assert payload["triage"]["triage_level"] == "EMERGENCY_ER"
    assert payload["consultation_needed"] is True
    assert payload["doctor_search"]["needs_location"] is True


@requires_model
def test_low_risk_text_stays_in_home_care():
    payload = client.post("/api/triage", json={"query_text": LOW_RISK_TEXT}).json()

    assert payload["triage"]["triage_level"] == "HOME_CARE_LIFESTYLE"
    assert payload["consultation_needed"] is False
    assert payload["doctor_search"]["needs_location"] is False


@requires_model
def test_structured_wizard_input_overrides_the_free_text():
    payload = client.post("/api/triage", json={
        "query_text": "I have some chest discomfort",
        "age": 67,
        "gender": "FEMALE",
        "systolic_bp": 188,
        "diastolic_bp": 122,
        "has_hypertension": True,
    }).json()

    assert payload["entities"]["age"] == 67
    assert payload["entities"]["gender"] == "FEMALE"
    assert payload["entities"]["systolic_bp"] == 188
    # A hypertensive crisis reading must reach the top escalation level.
    assert payload["triage"]["triage_level"] == "EMERGENCY_ER"


@requires_model
def test_triage_response_no_longer_claims_a_model_accuracy():
    payload = client.post("/api/triage", json={"query_text": LOW_RISK_TEXT}).json()
    # A single constant accuracy figure in every payload was misleading; metrics
    # now live behind /api/model-info with their baseline attached.
    assert "model_accuracy" not in payload["ml_result"]
    assert "model_accuracy" not in payload["triage"]


@requires_model
def test_triage_response_has_the_full_contract():
    payload = client.post("/api/triage", json={"query_text": EMERGENCY_TEXT}).json()
    assert {"status", "entities", "ml_result", "triage", "diet",
            "consultation_needed", "doctor_search"} <= set(payload.keys())
    assert payload["diet"]["protocol_name"]
    assert payload["diet"]["foods_to_eat"]
    assert payload["diet"]["foods_to_avoid"]
    assert payload["diet"]["sample_menu"]["Breakfast"]
    assert payload["diet"]["remedies"]


@requires_model
def test_risk_probability_is_a_percentage():
    payload = client.post("/api/triage", json={"query_text": LOW_RISK_TEXT}).json()
    assert 0.0 <= payload["ml_result"]["probability"] <= 100.0
    assert payload["ml_result"]["risk_category"]


@requires_model
def test_triage_advises_a_specialist_that_matches_the_triage_level():
    payload = client.post("/api/triage", json={"query_text": EMERGENCY_TEXT}).json()
    specialist = payload["doctor_search"]["recommended_specialist"]
    assert "mergency" in specialist
    assert payload["doctor_search"]["agency"]


# --------------------------------------------------------------------------
# Feature plumbing
# --------------------------------------------------------------------------

@requires_model
def test_cholesterol_is_not_inferred_from_hypertension():
    """Regression: cholesterol used to be derived from `has_hypertension`, so
    every hypertensive patient was silently handed an abnormal lipid panel they
    had never reported."""
    base = {"age": 58, "gender": "MALE", "height": 172, "weight": 88}
    without = client.post("/api/triage", json={**base, "has_hypertension": False}).json()
    with_htn = client.post("/api/triage", json={**base, "has_hypertension": True}).json()
    assert without["ml_result"]["probability"] == with_htn["ml_result"]["probability"]


@requires_model
def test_reported_cholesterol_reaches_the_model():
    base = {"age": 58, "gender": "MALE", "height": 172, "weight": 88,
            "systolic_bp": 150, "diastolic_bp": 92}
    normal = client.post("/api/triage", json={**base, "cholesterol": 1}).json()
    high = client.post("/api/triage", json={**base, "cholesterol": 3}).json()
    assert normal["ml_result"]["probability"] != high["ml_result"]["probability"]


@requires_model
def test_unknown_self_reported_values_still_produce_a_screening_result():
    """A patient who does not know their lab values is a normal case, not an
    error: the missing values must survive the round trip and still be scored."""
    payload = client.post("/api/triage", json={
        "age": 58, "gender": "MALE", "systolic_bp": 150, "diastolic_bp": 92,
        "cholesterol": None, "gluc": None, "alco": None, "active": None,
    }).json()
    assert 0.0 <= payload["ml_result"]["probability"] <= 100.0


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------

@requires_model
def test_simulator_reports_a_positive_reduction_for_better_vitals():
    payload = client.post("/api/simulate", json={
        "base_age": 54, "base_sys": 165, "base_dia": 100, "base_wt": 95, "base_ht": 172,
        "base_smk": True, "base_active": False,
        "target_sys": 120, "target_dia": 78, "target_wt": 76,
        "target_smk": False, "target_active": True,
    }).json()

    assert payload["risk_reduction"] > 0
    assert payload["target_risk"]["probability"] < payload["base_risk"]["probability"]


@requires_model
def test_simulator_does_not_leak_the_model_accuracy_constant():
    payload = client.post("/api/simulate", json={}).json()
    assert "model_accuracy" not in payload["base_risk"]
    assert "model_accuracy" not in payload["target_risk"]


@requires_model
def test_simulator_risk_values_are_bounded():
    payload = client.post("/api/simulate", json={}).json()
    for key in ("base_risk", "target_risk"):
        assert 0.0 <= payload[key]["probability"] <= 100.0
        assert payload[key]["theme"] in ("green", "yellow", "red", "blood-red")
