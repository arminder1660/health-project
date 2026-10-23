"""Tests for the rule-based clinical entity extractor."""
from src.entity_extraction import extract_clinical_entities


def test_age_is_read_from_a_sentence():
    assert extract_clinical_entities("I am 54 years old")["age"] == 54


def test_age_is_read_from_a_labelled_value():
    assert extract_clinical_entities("age: 71")["age"] == 71


def test_implausible_age_is_rejected():
    # 8 is below the 10-year floor the extractor accepts.
    assert extract_clinical_entities("I am 8 years old")["age"] is None


def test_gender_female_keywords():
    assert extract_clinical_entities("my wife has chest pain")["gender"] == "FEMALE"


def test_gender_male_keywords():
    assert extract_clinical_entities("my husband has chest pain")["gender"] == "MALE"


def test_blood_pressure_pair_is_parsed():
    entities = extract_clinical_entities("I am 58, my bp is 150/95")
    assert entities["systolic_bp"] == 150
    assert entities["diastolic_bp"] == 95


def test_hypertension_is_flagged_from_a_high_reading():
    assert extract_clinical_entities("bp 150/95")["has_hypertension"] is True


def test_hypertension_is_not_flagged_from_a_normal_reading():
    assert extract_clinical_entities("bp 118/76")["has_hypertension"] is False


def test_hypertension_is_flagged_from_the_word_alone():
    assert extract_clinical_entities("I take tablets for hypertension")["has_hypertension"] is True


def test_diabetes_keywords():
    assert extract_clinical_entities("I am diabetic since 2019")["has_diabetes"] is True


def test_smoker_is_detected():
    assert extract_clinical_entities("I smoke about 10 cigarettes a day")["is_smoker"] is True


def test_non_smoker_is_not_flagged():
    assert extract_clinical_entities("I am a non-smoker")["is_smoker"] is False


def test_quit_smoking_is_not_flagged():
    assert extract_clinical_entities("I quit smoking two years ago")["is_smoker"] is False


def test_family_history_needs_a_relative_and_a_cardiac_event():
    assert extract_clinical_entities("my father had a heart attack")["has_family_history"] is True


def test_family_history_absent_without_a_cardiac_event():
    assert extract_clinical_entities("my father lives in Pune")["has_family_history"] is False


def test_pain_character_pressure():
    assert "HEAVY_PRESSURE_TIGHTNESS" in extract_clinical_entities("heavy chest pressure")["pain_character"]


def test_pain_character_burning():
    assert "BURNING_ACID_REFLUX" in extract_clinical_entities("burning in my chest after food")["pain_character"]


def test_pain_character_palpitations():
    assert "PALPITATIONS_ARRHYTHMIA" in extract_clinical_entities("my heart is racing and fluttering")["pain_character"]


def test_pain_character_sharp():
    assert "SHARP_PLEURITIC_OR_MUSCULOSKELETAL" in extract_clinical_entities("sharp needle pricking pain")["pain_character"]


def test_radiation_to_left_arm():
    assert "LEFT_ARM_SHOULDER" in extract_clinical_entities("pain going down my left arm")["radiation"]


def test_radiation_to_jaw():
    assert "JAW_NECK_THROAT" in extract_clinical_entities("pain spreading to my jaw")["radiation"]


def test_trigger_exertion():
    assert "PHYSICAL_EXERTION" in extract_clinical_entities("it starts when I climb stairs")["triggers"]


def test_trigger_meal():
    assert "POST_PRANDIAL_MEALS" in extract_clinical_entities("it happens after a heavy meal")["triggers"]


def test_relief_by_rest():
    assert "RELIEVED_BY_REST" in extract_clinical_entities("it goes away with rest")["relief"]


def test_duration_seconds():
    assert extract_clinical_entities("it lasts a few seconds")["duration"] == "SECONDS"


def test_duration_minutes():
    assert extract_clinical_entities("it lasts about 5 minutes")["duration"] == "MINUTES_5_TO_15"


def test_duration_prolonged():
    assert extract_clinical_entities("it has been going on for 40 minutes")["duration"] == "PROLONGED_OVER_20_MINS"


def test_red_flag_cold_sweating():
    assert "COLD_SWEATING_DIAPHORESIS" in extract_clinical_entities("I broke into cold sweating")["red_flags"]


def test_red_flag_cold_sweating_is_negated():
    assert "COLD_SWEATING_DIAPHORESIS" not in extract_clinical_entities("there is no cold sweating")["red_flags"]


def test_red_flag_breathlessness():
    assert "ACUTE_DYSPNEA" in extract_clinical_entities("I feel breathless")["red_flags"]


def test_red_flag_breathlessness_is_negated():
    assert "ACUTE_DYSPNEA" not in extract_clinical_entities("no shortness of breath")["red_flags"]


def test_red_flag_syncope():
    assert "PRESYNCOPE_SYNCOPE" in extract_clinical_entities("I almost fainted")["red_flags"]


def test_peripheral_edema_goes_to_associated_symptoms():
    assert "PERIPHERAL_EDEMA" in extract_clinical_entities("my ankles are swollen")["associated_symptoms"]


def test_category_stable_angina():
    entities = extract_clinical_entities("crushing chest pressure when I climb stairs")
    assert entities["primary_symptom_category"] == "SUSPECTED_STABLE_ANGINA"


def test_category_acute_coronary_syndrome_when_prolonged():
    entities = extract_clinical_entities("heavy chest tightness for 45 minutes")
    assert entities["primary_symptom_category"] == "POTENTIAL_ACUTE_CORONARY_SYNDROME"


def test_category_reflux():
    entities = extract_clinical_entities("burning sour reflux after meals")
    assert entities["primary_symptom_category"] == "GASTROESOPHAGEAL_REFLUX_GERD"


def test_category_musculoskeletal():
    entities = extract_clinical_entities("sharp pricking pain when I press my ribs")
    assert entities["primary_symptom_category"] == "MUSCULOSKELETAL_CHEST_WALL_STRAIN"


def test_empty_text_returns_the_unknown_skeleton():
    entities = extract_clinical_entities("")
    assert entities["primary_symptom_category"] == "UNKNOWN"
    assert entities["age"] is None
    assert entities["pain_character"] == []


def test_output_contract_is_stable():
    expected_keys = {
        "raw_text", "age", "gender", "systolic_bp", "diastolic_bp",
        "has_hypertension", "has_diabetes", "is_smoker", "has_family_history",
        "pain_character", "radiation", "triggers", "relief", "duration",
        "red_flags", "associated_symptoms", "primary_symptom_category",
    }
    assert expected_keys <= set(extract_clinical_entities("anything").keys())
