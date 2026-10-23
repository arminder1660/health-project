"""Tests for the live doctor-discovery helpers.

Nothing here touches the network: the HTTP layer is monkeypatched, and the
ranking / filtering logic is exercised directly.
"""
import pytest

from src import geo_doctors
from src.geo_doctors import (
    _is_non_cardiac,
    _rank,
    _radius_viewbox,
    _relevance,
    geocode_place,
    haversine_distance,
    search_doctors_near,
)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def test_distance_to_the_same_point_is_zero():
    assert haversine_distance(32.7266, 74.8570, 32.7266, 74.8570) == 0.0


def test_distance_between_two_known_cities_is_plausible():
    # Jammu -> Delhi is roughly 500 km as the crow flies.
    distance = haversine_distance(32.7266, 74.8570, 28.6139, 77.2090)
    assert 430 < distance < 560


def test_distance_symmetry():
    forward = haversine_distance(32.7266, 74.8570, 19.0760, 72.8777)
    backward = haversine_distance(19.0760, 72.8777, 32.7266, 74.8570)
    assert forward == backward


def test_radius_viewbox_is_ordered_left_top_right_bottom():
    box = _radius_viewbox(32.7266, 74.8570, 15)
    left, top, right, bottom = (float(v) for v in box.split(","))
    assert left < right
    assert bottom < top
    # The box must actually straddle the centre point.
    assert left < 74.8570 < right
    assert bottom < 32.7266 < top


def test_radius_viewbox_grows_with_the_radius():
    small = _radius_viewbox(32.7266, 74.8570, 15)
    large = _radius_viewbox(32.7266, 74.8570, 80)
    small_left, small_top, small_right, _ = (float(v) for v in small.split(","))
    large_left, large_top, large_right, _ = (float(v) for v in large.split(","))
    assert (large_right - large_left) > (small_right - small_left)
    assert large_top > small_top


# --------------------------------------------------------------------------
# Safety filtering
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "Ayurvedic Panchakarma Centre",
    "Shri Homeopathy Clinic",
    "Unani Medical Centre",
    "Siddha Wellness Hospital",
    "Naturopathy and Yoga Therap Centre",
    "Smile Dental Clinic",
    "Netra Eye Hospital",
    "City Orthopaedic Centre",
    "Sai Veterinary Hospital",
])
def test_non_cardiac_providers_are_rejected(name):
    assert _is_non_cardiac(name, "", "CLINIC") is True


@pytest.mark.parametrize("name, facility_type", [
    ("City Heart Institute", "CARDIOLOGY_CENTRE"),
    ("Government Medical College Hospital", "HOSPITAL"),
    ("District Hospital", "HOSPITAL"),
    ("Sethi Cardiology Clinic", "CLINIC"),
])
def test_realistic_cardiac_destinations_are_kept(name, facility_type):
    assert _is_non_cardiac(name, "", facility_type) is False


def test_a_broad_hospital_keeps_its_place_despite_a_unit_in_its_name():
    # "Netralaya Multi Speciality Hospital" treats hearts as well as eyes.
    assert _is_non_cardiac("Netralaya Multi Speciality Hospital", "", "HOSPITAL") is False


def test_a_standalone_eye_clinic_does_not():
    assert _is_non_cardiac("Netralaya Eye Clinic", "", "CLINIC") is True


def test_alternative_medicine_speciality_tag_is_rejected():
    assert _is_non_cardiac("Wellness Centre", "ayurveda", "CLINIC") is True


def test_non_cardiac_speciality_tag_is_rejected():
    assert _is_non_cardiac("Wellness Centre", "ophthalmology", "CLINIC") is True


def test_diagnostic_lab_is_not_a_treatment_destination():
    assert _is_non_cardiac("City Diagnostic Centre", "", "CLINIC") is True


# --------------------------------------------------------------------------
# Relevance + ranking
# --------------------------------------------------------------------------

def test_cardiology_outranks_a_generic_clinic():
    cardiac = _relevance("City Cardiac Centre", "CARDIOLOGY_CENTRE", {}, "SPECIALIST_CARDIOLOGIST")
    generic = _relevance("Family Clinic", "CLINIC", {}, "SPECIALIST_CARDIOLOGIST")
    assert cardiac > generic


def test_emergency_triage_prefers_a_hospital_over_a_clinic():
    hospital = _relevance("District Hospital", "HOSPITAL", {}, "EMERGENCY_ER")
    clinic = _relevance("Family Clinic", "CLINIC", {}, "EMERGENCY_ER")
    assert hospital > clinic


def test_an_emergency_room_tag_raises_relevance():
    plain = _relevance("District Hospital", "HOSPITAL", {}, None)
    with_er = _relevance("District Hospital", "HOSPITAL", {"emergency": "yes"}, None)
    assert with_er > plain


def make_facility(name, lat, lon, distance_km, relevance=0.0):
    return {
        "name": name, "lat": lat, "lon": lon,
        "distance_km": distance_km, "relevance": relevance,
    }


def test_ranking_drops_facilities_outside_the_radius():
    items = [
        make_facility("Near Hospital", 32.70, 74.85, 4.0),
        make_facility("Far Hospital", 32.10, 74.10, 120.0),
    ]
    kept = _rank(items, 32.7266, 74.8570, 40, 10)
    assert [f["name"] for f in kept] == ["Near Hospital"]


def test_ranking_prefers_the_closer_facility_when_relevance_is_equal():
    items = [
        make_facility("Distant Hospital", 32.90, 75.10, 30.0),
        make_facility("Close Hospital", 32.74, 74.86, 2.0),
    ]
    kept = _rank(items, 32.7266, 74.8570, 40, 10)
    assert [f["name"] for f in kept] == ["Close Hospital", "Distant Hospital"]


def test_relevance_can_outweigh_a_modest_distance_difference():
    items = [
        make_facility("Generic Clinic", 32.74, 74.86, 3.0, relevance=20.0),
        make_facility("Cardiac Centre", 32.80, 74.90, 9.0, relevance=140.0),
    ]
    kept = _rank(items, 32.7266, 74.8570, 40, 10)
    assert kept[0]["name"] == "Cardiac Centre"


def test_ranking_removes_duplicates():
    items = [
        make_facility("City Hospital", 32.7266, 74.8570, 0.5),
        make_facility("city hospital", 32.7267, 74.8571, 0.6),
    ]
    assert len(_rank(items, 32.7266, 74.8570, 40, 10)) == 1


def test_ranking_honours_the_limit():
    items = [make_facility(f"Hospital {i}", 32.70 + i / 1000, 74.85, 1.0 + i) for i in range(20)]
    assert len(_rank(items, 32.7266, 74.8570, 40, 5)) == 5


def test_ranking_removes_the_internal_relevance_key():
    items = [make_facility("City Hospital", 32.7266, 74.8570, 1.0, relevance=60.0)]
    kept = _rank(items, 32.7266, 74.8570, 40, 10)
    assert "relevance" not in kept[0]


def test_ranking_fills_in_the_distance_text():
    items = [make_facility("City Hospital", 32.7266, 74.8570, 1.5)]
    assert _rank(items, 32.7266, 74.8570, 40, 10)[0]["distance_text"] == "1.5 km away"


# --------------------------------------------------------------------------
# Geocoding (HTTP monkeypatched)
# --------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def test_geocode_place_parses_a_nominatim_hit(monkeypatch):
    payload = [{
        "lat": "32.7266", "lon": "74.8570",
        "display_name": "Jammu, Jammu and Kashmir, India",
        "address": {"city": "Jammu", "state": "Jammu and Kashmir", "country": "India"},
    }]
    monkeypatch.setattr(geo_doctors.requests, "get", lambda *a, **k: FakeResponse(payload))

    result = geocode_place("Jammu-test-a")
    assert result["lat"] == pytest.approx(32.7266)
    assert result["lon"] == pytest.approx(74.8570)
    assert result["city"] == "Jammu"


def test_geocode_place_returns_none_for_an_unknown_place(monkeypatch):
    monkeypatch.setattr(geo_doctors.requests, "get", lambda *a, **k: FakeResponse([]))
    assert geocode_place("Nowhereville-test-b") is None


def test_geocode_place_survives_a_network_error(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(geo_doctors.requests, "get", boom)
    assert geocode_place("Offline-test-c") is None


def test_geocode_place_survives_a_non_200_response(monkeypatch):
    monkeypatch.setattr(geo_doctors.requests, "get", lambda *a, **k: FakeResponse({}, 429))
    assert geocode_place("Throttled-test-d") is None


def test_geocode_place_is_cached(monkeypatch):
    calls = []

    def counting_get(*a, **k):
        calls.append(1)
        return FakeResponse([{
            "lat": "1.0", "lon": "2.0", "display_name": "X", "address": {"city": "X"},
        }])

    monkeypatch.setattr(geo_doctors.requests, "get", counting_get)
    geocode_place("Cachetest-e")
    geocode_place("Cachetest-e")
    assert len(calls) == 1


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def test_no_location_asks_the_patient_for_one():
    result = search_doctors_near()
    assert result["status"] == "error"
    assert result["doctors"] == []
    assert "city" in result["message"].lower()


def test_an_unresolvable_place_returns_an_actionable_message(monkeypatch):
    monkeypatch.setattr(geo_doctors, "geocode_place", lambda place: None)
    result = search_doctors_near(location="Nowhereville-xyz-999")
    assert result["status"] == "error"
    assert result["doctors"] == []
    assert "Nowhereville-xyz-999" in result["message"]


def test_coordinates_skip_geocoding_and_query_the_providers(monkeypatch):
    seen = {}

    def fake_nominatim(lat, lon, radius_km, triage_level):
        seen["lat"] = lat
        return []

    monkeypatch.setattr(geo_doctors, "search_nominatim", fake_nominatim)
    monkeypatch.setattr(geo_doctors, "search_photon", lambda *a, **k: ([], True))
    monkeypatch.setattr(geo_doctors, "search_overpass", lambda *a, **k: [])

    result = search_doctors_near(lat=32.7266, lon=74.8570, radius_km=15)
    assert seen["lat"] == pytest.approx(32.7266)
    assert result["status"] == "error"
    assert result["doctors"] == []


def test_a_successful_search_returns_real_shaped_records(monkeypatch):
    facility = {
        "id": "osm-node-1",
        "name": "Government Medical College Hospital",
        "specialty": "Hospital",
        "facility_type": "HOSPITAL",
        "city": "Jammu",
        "lat": 32.7300,
        "lon": 74.8600,
        "address": "Bakshi Nagar, Jammu",
        "contact": "+911912580000",
        "website": "https://example.org",
        "google_maps": "https://www.google.com/maps/search/?api=1&query=32.73,74.86",
        "profile_url": "https://www.openstreetmap.org/node/1",
        "photo_initials": "GM",
        "distance_km": 0.6,
        "distance_text": "600 m away",
        "source": "OpenStreetMap",
        "relevance": 60.0,
    }
    monkeypatch.setattr(geo_doctors, "search_nominatim", lambda *a, **k: [dict(facility)])
    # A single Nominatim hit falls below the widening threshold, so the Photon
    # and Overpass fallbacks would otherwise go to the live network.
    monkeypatch.setattr(geo_doctors, "search_photon", lambda *a, **k: ([], True))
    monkeypatch.setattr(geo_doctors, "search_overpass", lambda *a, **k: [])

    result = search_doctors_near(lat=32.7266, lon=74.8570, radius_km=15)
    assert result["status"] == "success"
    assert result["total_facilities"] == 1

    doctor = result["doctors"][0]
    # Provenance is mandatory: a reviewer must be able to verify the place.
    assert doctor["source"] == "OpenStreetMap"
    assert doctor["profile_url"].startswith("https://")
    assert doctor["name"] == "Government Medical College Hospital"


def test_only_facilities_inside_the_radius_survive(monkeypatch):
    def fake_nominatim(lat, lon, radius_km, triage_level):
        return [
            dict(id="a", name="Close Hospital", specialty="Hospital", facility_type="HOSPITAL",
                 city="Jammu", lat=lat + 0.01, lon=lon, address="", contact="", website="",
                 google_maps="", profile_url="", photo_initials="CH", distance_km=1.2,
                 distance_text="", source="OpenStreetMap", relevance=60.0),
            dict(id="b", name="Very Far Hospital", specialty="Hospital", facility_type="HOSPITAL",
                 city="Delhi", lat=lat + 5.0, lon=lon, address="", contact="", website="",
                 google_maps="", profile_url="", photo_initials="VF", distance_km=556.0,
                 distance_text="", source="OpenStreetMap", relevance=60.0),
        ]

    monkeypatch.setattr(geo_doctors, "search_nominatim", fake_nominatim)
    monkeypatch.setattr(geo_doctors, "search_photon", lambda *a, **k: ([], True))
    monkeypatch.setattr(geo_doctors, "search_overpass", lambda *a, **k: [])

    result = search_doctors_near(lat=32.7266, lon=74.8570, radius_km=15)
    names = [d["name"] for d in result["doctors"]]
    assert "Close Hospital" in names
    assert "Very Far Hospital" not in names


def test_results_report_the_radius_actually_used(monkeypatch):
    monkeypatch.setattr(geo_doctors, "search_nominatim", lambda *a, **k: [])
    monkeypatch.setattr(geo_doctors, "search_photon", lambda *a, **k: ([], True))
    monkeypatch.setattr(geo_doctors, "search_overpass", lambda *a, **k: [])

    result = search_doctors_near(lat=32.7266, lon=74.8570)
    # With nothing found the search escalates to the widest ring.
    assert result["search_radius_km"] == geo_doctors.RADIUS_LADDER_KM[-1]
