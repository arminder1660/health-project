"""CardiaNova - Live Doctor Discovery by Patient Location.

Finds real cardiologists, cardiac centres, hospitals and clinics near the
place a patient says they are located in, using live internet data:

  * Default  : OpenStreetMap -- Nominatim geocodes the city and then performs a
               bounded category search (hospital / clinic / doctor) inside a
               radius around it. Completely free: no API key, no signup, no
               credit card.
  * Backup   : Photon (same OpenStreetMap data) picks up anything the category
               search missed, and Overpass API is the final deeper sweep used
               for small towns where the first two sources come up short.
  * Optional : Google Places API -- used automatically when the environment
               variable GOOGLE_PLACES_API_KEY is present. Returns richer data
               such as individual doctor names, star ratings and review counts.

Nothing is fabricated here. Every returned entry comes from a live geographic
database and carries a ``source`` plus a ``profile_url`` so the patient (or a
reviewer) can verify the place really exists. Providers that cannot actually
investigate a heart (ayurveda, homoeopathy, unani, siddha, naturopathy,
chiropractic) are filtered out, because listing them for suspected cardiac
disease is a patient-safety problem, not a convenience.
"""
from __future__ import annotations

import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USER_AGENT = "CardiaNova-HeartScreening/2.0 (patient triage demo)"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL = "https://photon.komoot.io/api/"

# Tried in order; the public Overpass instances are often at capacity, so
# having spares keeps the feature alive without a user-visible failure.
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_PLACES_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"
GOOGLE_FIELD_MASK = ",".join((
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.rating",
    "places.userRatingCount",
    "places.internationalPhoneNumber",
    "places.websiteUri",
    "places.googleMapsUri",
    "places.currentOpeningHours",
    "places.primaryTypeDisplayName",
))

# OpenStreetMap tags that count as "a place a heart patient can go".
HEALTH_OSM_TAGS = (
    "amenity:hospital", "amenity:clinic", "amenity:doctors",
    "healthcare:hospital", "healthcare:centre", "healthcare:doctor",
)

# Widening search rings: a metro area resolves on the first ring, while a
# small town automatically expands until it finds real facilities.
RADIUS_LADDER_KM = (15, 40, 80)
MIN_RESULTS_BEFORE_WIDENING = 6

HTTP_TIMEOUT = 15
PHOTON_TIMEOUT = 8
# Wide enough to allow a retry of a dropped connection: 8s + pause + 8s.
PHOTON_TIME_BUDGET = 18.0
AMBULANCE_NUMBER = "108"

_CARDIAC_PATTERN = re.compile(r"cardio|heart|cardiac", re.IGNORECASE)

# A place named after one of these is not where a heart patient should be sent.
# Anchored at a word start, so "Eye Hospital"/"Dental Clinic" match while the
# similarly-spelled surname "Peterson" does not.
_NON_CARDIAC_NAME_RE = re.compile(
    r"(?:"
    r"\b(?:netralaya|ophthalm|eye|dental|dentist|orthodont|orthop|derma|skin|"
    r"psych|gyn|ivf|fertility|matern|cosmet|physio|optical|hearing)"
    r"|"
    r"\b(?:hair|ortho|ent|veterinar\w*|vet|pet)\b"
    r")",
    re.IGNORECASE,
)

# Alternative-medicine providers that must NOT be offered when cardiac care is
# advised. They cannot run an ECG, troponin assay or cath-lab procedure, so
# listing them next to a suspected cardiac event is a patient-safety problem.
# The stems end in \w* so both "Homeopathy" and "Homeopathic" match, but the
# ambiguous ones keep a hard word boundary so a surname such as "Siddharth"
# is not mistaken for a Siddha practitioner.
_ALTERNATIVE_MEDICINE_RE = re.compile(
    r"\b(?:ayurved\w*|homeopath\w*|homoeopath\w*|unani\b|siddha\b|naturopath\w*|"
    r"acupunct\w*|chiropract\w*|reiki\b|yoga\s+therap\w*|panchakarma\b)",
    re.IGNORECASE,
)

# Specialty tags from OpenStreetMap that mean the same thing.
_ALTERNATIVE_MEDICINE_TAGS = (
    "ayurveda", "homeopathy", "homoeopathy", "unani", "siddha",
    "naturopathy", "alternative", "acupuncture", "chiropractic",
)

# These names describe a diagnostic lab rather than a place that treats patients.
_DIAGNOSTIC_ONLY_RE = re.compile(
    r"\b(?:diagnostic|patholog|pathlab|scan|imaging|radiology|blood bank|laborator)\b",
    re.IGNORECASE,
)

# A hospital that openly says it is multi-speciality is still a valid destination,
# even when an eye or dental unit happens to be named in the title.
_BROAD_HOSPITAL_RE = re.compile(r"multi|general|super special|medical college|institute",
                                re.IGNORECASE)

# Speciality tags that mean "this is not where a heart patient should go".
_NON_CARDIAC_SPECIALITIES = (
    "ophthalmolog", "eye", "dental", "dentist", "orthodont",
    "dermatolog", "psychiatr", "psycholog", "veterinar", "podiatr",
)


def _load_env_file() -> None:
    """Reads a local .env file so a Google key can be added without code edits."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and name not in os.environ:
                os.environ[name] = value
    except OSError:
        pass


_load_env_file()


def google_api_key() -> str:
    """The Google Places API key, or an empty string when not configured."""
    return os.getenv("GOOGLE_PLACES_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# Geometry and formatting helpers
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS coordinates, in kilometres."""
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return round(radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)


def _distance_text(dist_km: Optional[float]) -> str:
    if dist_km is None:
        return ""
    if dist_km < 1:
        return f"{int(round(dist_km * 1000))} m away"
    return f"{dist_km} km away"


def _initials(name: str) -> str:
    words = [w for w in re.split(r"[\s,.&'\-]+", name) if w]
    meaningful = [w for w in words if w.lower() not in ("dr", "the", "and", "of", "hospital", "clinic")]
    if len(meaningful) >= 2:
        return (meaningful[0][0] + meaningful[1][0]).upper()
    if len(meaningful) == 1:
        return meaningful[0][:2].upper()
    return words[0][:2].upper() if words else "DR"


def _maps_link(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"


def _is_non_cardiac(name: str, specialty_tag: str, facility_type: str) -> bool:
    """Filters out eye/dental/physio style places that nobody asked for."""
    if specialty_tag and any(bad in specialty_tag for bad in _NON_CARDIAC_SPECIALITIES):
        return True
    # Alternative-medicine providers are never a destination for suspected
    # cardiac disease, regardless of how the facility is categorised.
    if _ALTERNATIVE_MEDICINE_RE.search(name):
        return True
    if specialty_tag and any(bad in specialty_tag for bad in _ALTERNATIVE_MEDICINE_TAGS):
        return True
    if _NON_CARDIAC_NAME_RE.search(name):
        # "Netralaya Multi Speciality Hospital" is still a valid destination,
        # so a broad hospital keeps its place; a standalone eye clinic does not.
        return not (facility_type == "HOSPITAL" and _BROAD_HOSPITAL_RE.search(name))
    if facility_type in ("CLINIC", "GENERAL_PHYSICIAN") and _DIAGNOSTIC_ONLY_RE.search(name):
        return True
    return False


# ---------------------------------------------------------------------------
# Geocoding: turn "Jammu" into coordinates
# ---------------------------------------------------------------------------

_GEOCODE_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


def _geocode_nominatim(place: str) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": place, "format": "jsonv2", "limit": 1, "addressdetails": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        hits = resp.json()
        if not hits:
            return None
        hit = hits[0]
        address = hit.get("address") or {}
        city = (
            address.get("city") or address.get("town") or address.get("village")
            or address.get("municipality") or address.get("state_district")
            or address.get("county") or place
        )
        return {
            "lat": float(hit["lat"]),
            "lon": float(hit["lon"]),
            "display_name": hit.get("display_name", place),
            "city": city,
            "state": address.get("state", ""),
            "country": address.get("country", ""),
        }
    except Exception as exc:
        print(f"[geo_doctors] Nominatim geocode failed for {place!r}: {exc}")
        return None


def _geocode_google(place: str) -> Optional[Dict[str, Any]]:
    key = google_api_key()
    if not key:
        return None
    try:
        resp = requests.get(GOOGLE_GEOCODE_URL, params={"address": place, "key": key},
                            timeout=HTTP_TIMEOUT)
        results = (resp.json() or {}).get("results") or []
        if not results:
            return None
        top = results[0]
        components = top.get("address_components") or []
        locality = next(
            (c["long_name"] for c in components
             if {"locality", "administrative_area_level_3", "administrative_area_level_2"}
             & set(c.get("types", []))),
            place,
        )
        location = top["geometry"]["location"]
        return {
            "lat": float(location["lat"]),
            "lon": float(location["lng"]),
            "display_name": top.get("formatted_address", place),
            "city": locality,
            "state": "",
            "country": "",
        }
    except Exception as exc:
        print(f"[geo_doctors] Google geocode failed for {place!r}: {exc}")
        return None


def geocode_place(place: str) -> Optional[Dict[str, Any]]:
    """Resolves a free-text place name to coordinates (cached per session)."""
    cache_key = place.strip().lower()
    if not cache_key:
        return None
    if cache_key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[cache_key]
    resolved = _geocode_nominatim(place) or _geocode_google(place)
    _GEOCODE_CACHE[cache_key] = resolved
    return resolved


# ---------------------------------------------------------------------------
# Provider 1: Nominatim category search (OpenStreetMap, free, fast, no key)
# ---------------------------------------------------------------------------

NOMINATIM_CATEGORY_TERMS = ("hospital", "clinic", "doctor")


def _radius_viewbox(lat: float, lon: float, radius_km: float) -> str:
    """A 'left,top,right,bottom' box covering the search radius around a point."""
    d_lat = radius_km / 111.0
    d_lon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.02))
    return f"{lon - d_lon:.5f},{lat + d_lat:.5f},{lon + d_lon:.5f},{lat - d_lat:.5f}"


def _nominatim_fetch(term: str, viewbox: str) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": term, "viewbox": viewbox, "bounded": 1, "format": "jsonv2",
                    "limit": 40, "addressdetails": 1, "extratags": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            print(f"[geo_doctors] Nominatim {term!r} -> HTTP {resp.status_code}")
            return []
        return resp.json() or []
    except Exception as exc:
        print(f"[geo_doctors] Nominatim {term!r} failed: {exc}")
        return []


def _nominatim_item(hit: Dict[str, Any], lat: float, lon: float,
                    triage_level: Optional[str]) -> Optional[Dict[str, Any]]:
    """Turns one Nominatim result into the facility shape the UI renders."""
    try:
        el_lat, el_lon = float(hit["lat"]), float(hit["lon"])
    except (KeyError, TypeError, ValueError):
        return None

    display = hit.get("display_name") or ""
    name = (hit.get("name") or display.split(",")[0]).strip()
    if not name:
        return None

    category = (hit.get("type") or "").lower()
    if category == "hospital":
        facility_type, label = "HOSPITAL", "Hospital"
    elif category == "doctors":
        facility_type, label = "GENERAL_PHYSICIAN", "General Physician / Doctor"
    else:
        facility_type, label = "CLINIC", "Clinic / Medical Centre"
    if _CARDIAC_PATTERN.search(name):
        facility_type, label = "CARDIOLOGY_CENTRE", "Cardiac Care Centre"

    extra = hit.get("extratags") or {}
    if _is_non_cardiac(name, (extra.get("healthcare:speciality") or "").lower(), facility_type):
        return None

    address = hit.get("address") or {}
    place_text = ", ".join(str(p) for p in (
        address.get("house_number"), address.get("road"),
        address.get("suburb") or address.get("neighbourhood"),
        address.get("city") or address.get("town") or address.get("village"),
        address.get("postcode"),
    ) if p) or display

    osm_type = hit.get("osm_type") or "node"
    osm_id = hit.get("osm_id")
    dist_km = haversine_distance(lat, lon, el_lat, el_lon)

    return {
        "id": f"osm-{osm_type}-{osm_id}",
        "name": name,
        "specialty": label,
        "facility_type": facility_type,
        "hospital": name,
        "city": (address.get("city") or address.get("town")
                 or address.get("village") or ""),
        "lat": el_lat,
        "lon": el_lon,
        "address": place_text,
        "contact": extra.get("phone") or extra.get("contact:phone") or "",
        "emergency_phone": AMBULANCE_NUMBER,
        "experience": "",
        "rating": "",
        "timings": (f"OPD: {extra['opening_hours']}" if extra.get("opening_hours")
                    else "Call ahead to confirm OPD hours"),
        "website": extra.get("website") or extra.get("contact:website") or "",
        "google_maps": _maps_link(el_lat, el_lon),
        "profile_url": f"https://www.openstreetmap.org/{osm_type}/{osm_id}",
        "photo_initials": _initials(name),
        "distance_km": dist_km,
        "distance_text": _distance_text(dist_km),
        "source": "OpenStreetMap",
        "relevance": _relevance(name, facility_type, extra, triage_level),
    }


def search_nominatim(lat: float, lon: float, radius_km: int,
                     triage_level: Optional[str]) -> List[Dict[str, Any]]:
    """Finds real facilities inside one search ring around a point.

    Nominatim asks clients to stay under one request per second, so the three
    category queries are spaced out, which costs about three seconds in total.
    """
    viewbox = _radius_viewbox(lat, lon, radius_km)
    items: List[Dict[str, Any]] = []

    for index, term in enumerate(NOMINATIM_CATEGORY_TERMS):
        if index:
            time.sleep(1.1)
        for hit in _nominatim_fetch(term, viewbox):
            item = _nominatim_item(hit, lat, lon, triage_level)
            if item:
                items.append(item)
    return items


# ---------------------------------------------------------------------------
# Ranking shared by every provider
# ---------------------------------------------------------------------------

def _relevance(name: str, facility_type: str, tags: Dict[str, Any],
               triage_level: Optional[str]) -> float:
    speciality = (tags.get("healthcare:speciality") or "").lower()
    score = 0.0
    if _CARDIAC_PATTERN.search(speciality):
        score += 100
    if _CARDIAC_PATTERN.search(name):
        score += 60
    if facility_type == "HOSPITAL":
        score += 40
    elif facility_type == "CLINIC":
        score += 25
    elif facility_type == "GENERAL_PHYSICIAN":
        score += 20
    if str(tags.get("emergency", "")).lower() in ("yes", "emergency_room"):
        score += 30

    # Nudge the ranking according to what the triage engine advised.
    if triage_level == "EMERGENCY_ER":
        score += 30 if facility_type == "HOSPITAL" else -25
    elif triage_level == "SPECIALIST_CARDIOLOGIST":
        score += 25 if facility_type == "CARDIOLOGY_CENTRE" else 0
    elif triage_level == "GENERAL_PHYSICIAN":
        score += 15 if facility_type in ("CLINIC", "GENERAL_PHYSICIAN") else 0
    return score


def _rank(items: List[Dict[str, Any]], lat: float, lon: float, radius_km: float,
          limit: int) -> List[Dict[str, Any]]:
    """Drops anything outside the radius, removes duplicates, orders by relevance.

    Relevance alone would let a 70 km hospital outrank a good one down the road,
    so the ranking is tempered by distance -- far closer beats slightly better.
    """
    seen: set = set()
    kept: List[Tuple[float, float, Dict[str, Any]]] = []

    for item in items:
        dist = item.get("distance_km")
        if dist is None or dist > radius_km:
            continue
        key = (item["name"].lower(), round(item["lat"], 3), round(item["lon"], 3))
        if key in seen:
            continue
        seen.add(key)
        item["distance_text"] = _distance_text(dist)
        score = item.pop("relevance", 0.0) - min(dist, 60.0) * 0.8
        kept.append((score, dist, item))

    kept.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in kept[:limit]]


# ---------------------------------------------------------------------------
# Provider 2: Photon (OpenStreetMap data, free, no key)
# ---------------------------------------------------------------------------

_PHOTON_QUERIES = (
    ("hospital", ("amenity:hospital", "healthcare:hospital")),
    ("clinic doctor", ("amenity:clinic", "amenity:doctors", "healthcare:centre", "healthcare:doctor")),
    ("cardiology heart centre", HEALTH_OSM_TAGS),
)


def _photon_fetch(term: str, lat: float, lon: float) -> List[Dict[str, Any]]:
    """One Photon query. Photon occasionally drops a parallel connection, so a
    single quick retry keeps a whole search from losing its best query."""
    params = {"q": term, "lat": lat, "lon": lon, "limit": 25, "lang": "en",
              "osm_tag": list(HEALTH_OSM_TAGS)}
    for attempt in range(2):
        try:
            resp = requests.get(PHOTON_URL, params=params,
                                headers={"User-Agent": USER_AGENT},
                                timeout=PHOTON_TIMEOUT)
            if resp.status_code == 200:
                return resp.json().get("features") or []
            print(f"[geo_doctors] Photon query {term!r} -> HTTP {resp.status_code}")
        except Exception as exc:
            print(f"[geo_doctors] Photon query {term!r} failed: {exc}")
        if attempt == 0:
            time.sleep(1.0)
    return []


def search_photon(city: str, lat: float, lon: float,
                  triage_level: Optional[str]) -> Tuple[List[Dict[str, Any]], bool]:
    """Searches Photon for health facilities around a city.

    The queries are issued in parallel, so the patient waits for the slowest
    single call rather than the sum of all of them. Returns ``(items, reached)``
    where ``reached`` is False only when every request failed on the network.
    """
    deadline = time.monotonic() + PHOTON_TIME_BUDGET
    items: List[Dict[str, Any]] = []
    seen_osm_ids: set = set()
    reached = False

    with ThreadPoolExecutor(max_workers=len(_PHOTON_QUERIES)) as pool:
        # Including the city in the search text sharply improves local relevance.
        futures = [pool.submit(_photon_fetch, f"{term} {city}".strip(), lat, lon)
                   for term, _tags in _PHOTON_QUERIES]

        for future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                continue
            try:
                features = future.result(timeout=remaining)
                reached = True
            except Exception as exc:
                print(f"[geo_doctors] Photon query timed out: {exc}")
                continue

            for feature in features:
                props = feature.get("properties") or {}
                coords = (feature.get("geometry") or {}).get("coordinates") or []
                if len(coords) != 2:
                    continue
                el_lon, el_lat = float(coords[0]), float(coords[1])

                osm_id = props.get("osm_id")
                osm_type = props.get("osm_type") or "node"
                unique = f"{osm_type}/{osm_id}"
                if unique in seen_osm_ids:
                    continue
                seen_osm_ids.add(unique)

                value = (props.get("osm_value") or "").lower()
                name = (props.get("name") or "").strip()
                if not name:
                    # Fall back to a street-based label so the entry is still useful.
                    name = ", ".join(p for p in (props.get("street"), props.get("city")) if p)
                if not name:
                    continue

                if value == "hospital":
                    facility_type, label = "HOSPITAL", "Hospital"
                elif _CARDIAC_PATTERN.search(name):
                    facility_type, label = "CARDIOLOGY_CENTRE", "Cardiac Care Centre"
                elif value == "doctors":
                    facility_type, label = "GENERAL_PHYSICIAN", "General Physician / Doctor"
                else:
                    facility_type, label = "CLINIC", "Clinic / Medical Centre"

                if _is_non_cardiac(name, "", facility_type):
                    continue

                dist_km = haversine_distance(lat, lon, el_lat, el_lon)
                address = ", ".join(str(p) for p in (
                    props.get("housenumber"), props.get("street"), props.get("city"),
                    props.get("postcode"),
                ) if p) or city

                items.append({
                    "id": f"osm-{osm_type}-{osm_id}",
                    "name": name,
                    "specialty": label,
                    "facility_type": facility_type,
                    "hospital": name,
                    "city": props.get("city") or props.get("district") or city,
                    "lat": el_lat,
                    "lon": el_lon,
                    "address": address,
                    "contact": "",
                    "emergency_phone": AMBULANCE_NUMBER,
                    "experience": "",
                    "rating": "",
                    "timings": "Call ahead to confirm OPD hours",
                    "website": "",
                    "google_maps": _maps_link(el_lat, el_lon),
                    "profile_url": f"https://www.openstreetmap.org/{osm_type}/{osm_id}",
                    "photo_initials": _initials(name),
                    "distance_km": dist_km,
                    "distance_text": _distance_text(dist_km),
                    "source": "OpenStreetMap",
                    "relevance": _relevance(name, facility_type, {}, triage_level),
                })

    return items, reached


# ---------------------------------------------------------------------------
# Provider 3: Overpass (deeper tag-based OSM scan, used as a last resort)
# ---------------------------------------------------------------------------

def _overpass_query(query: str) -> List[Dict[str, Any]]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for url in OVERPASS_ENDPOINTS:
        try:
            resp = requests.post(url, data={"data": query}, headers=headers,
                                 timeout=HTTP_TIMEOUT + 20)
            if resp.status_code == 200:
                return resp.json().get("elements") or []
            print(f"[geo_doctors] Overpass {url} -> HTTP {resp.status_code}")
        except Exception as exc:
            print(f"[geo_doctors] Overpass {url} failed: {exc}")
    return []


def search_overpass(lat: float, lon: float, radius_km: int, city: str,
                    triage_level: Optional[str]) -> List[Dict[str, Any]]:
    radius_m = radius_km * 1000
    query = f"""[out:json][timeout:{HTTP_TIMEOUT}];
(
  nwr(around:{radius_m},{lat},{lon})[amenity=hospital];
  nwr(around:{radius_m},{lat},{lon})[amenity=clinic];
  nwr(around:{radius_m},{lat},{lon})[amenity=doctors];
  nwr(around:{radius_m},{lat},{lon})[healthcare=hospital];
  nwr(around:{radius_m},{lat},{lon})[healthcare=centre];
  nwr(around:{radius_m},{lat},{lon})[healthcare=doctor];
);
out center 250;"""

    items: List[Dict[str, Any]] = []
    for element in _overpass_query(query):
        tags = element.get("tags") or {}
        name = (tags.get("name") or "").strip()
        if not name:
            continue

        if "lat" in element:
            el_lat, el_lon = float(element["lat"]), float(element["lon"])
            osm_type = "node"
        else:
            center = element.get("center") or {}
            if "lat" not in center:
                continue
            el_lat, el_lon = float(center["lat"]), float(center["lon"])
            osm_type = "way"

        speciality = (tags.get("healthcare:speciality") or "").lower()
        if any(bad in speciality for bad in _NON_CARDIAC_SPECIALITIES):
            continue

        amenity = (tags.get("amenity") or "").lower()
        healthcare = (tags.get("healthcare") or "").lower()
        if amenity == "hospital" or healthcare == "hospital":
            facility_type, label = "HOSPITAL", "Hospital"
        elif _CARDIAC_PATTERN.search(name) or _CARDIAC_PATTERN.search(speciality):
            facility_type, label = "CARDIOLOGY_CENTRE", "Cardiac Care Centre"
        elif amenity == "doctors" or healthcare == "doctor":
            facility_type, label = "GENERAL_PHYSICIAN", "General Physician / Doctor"
        else:
            facility_type, label = "CLINIC", "Clinic / Medical Centre"

        if _is_non_cardiac(name, speciality, facility_type):
            continue

        addr_parts = (tags.get("addr:housenumber"), tags.get("addr:street"),
                      tags.get("addr:suburb"), tags.get("addr:city") or tags.get("addr:town"),
                      tags.get("addr:postcode"))
        address = tags.get("addr:full") or ", ".join(p for p in addr_parts if p) or city
        opening = tags.get("opening_hours") or ""
        osm_id = element.get("id")

        items.append({
            "id": f"osm-{osm_type}-{osm_id}",
            "name": name,
            "specialty": label,
            "facility_type": facility_type,
            "hospital": name,
            "city": city,
            "lat": el_lat,
            "lon": el_lon,
            "address": address,
            "contact": tags.get("phone") or tags.get("contact:phone") or "",
            "emergency_phone": AMBULANCE_NUMBER,
            "experience": "",
            "rating": "",
            "timings": f"OPD: {opening}" if opening else "Call ahead to confirm OPD hours",
            "website": tags.get("website") or tags.get("contact:website") or "",
            "google_maps": _maps_link(el_lat, el_lon),
            "profile_url": f"https://www.openstreetmap.org/{osm_type}/{osm_id}",
            "photo_initials": _initials(name),
            "distance_km": haversine_distance(lat, lon, el_lat, el_lon),
            "distance_text": "",
            "source": "OpenStreetMap",
            "relevance": _relevance(name, facility_type, tags, triage_level),
        })
    return items


# ---------------------------------------------------------------------------
# Provider 4: Google Places (only when a key is configured)
# ---------------------------------------------------------------------------

def search_google_places(key: str, lat: float, lon: float, radius_km: int,
                         limit: int, location_label: str) -> List[Dict[str, Any]]:
    body = {
        "textQuery": f"cardiologist near {location_label}",
        "maxResultCount": max(5, min(limit, 20)),
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": float(radius_km * 1000),
            }
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": GOOGLE_FIELD_MASK,
    }
    resp = requests.post(GOOGLE_PLACES_TEXT_URL, json=body, headers=headers, timeout=HTTP_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    items: List[Dict[str, Any]] = []
    for place in resp.json().get("places", []):
        name = (place.get("displayName") or {}).get("text") or ""
        location = place.get("location") or {}
        p_lat, p_lon = location.get("latitude"), location.get("longitude")
        if not name or p_lat is None or p_lon is None:
            continue

        rating_value = place.get("rating")
        review_count = place.get("userRatingCount")
        if rating_value and review_count:
            rating = f"{rating_value} \u2605 ({review_count:,} Google reviews)"
        elif rating_value:
            rating = f"{rating_value} \u2605"
        else:
            rating = ""

        opening = (place.get("currentOpeningHours") or {}).get("weekdayDescriptions") or []
        maps_url = place.get("googleMapsUri") or _maps_link(p_lat, p_lon)
        dist_km = haversine_distance(lat, lon, p_lat, p_lon)

        items.append({
            "id": f"google-{abs(hash((name, round(p_lat, 4))))}",
            "name": name,
            "specialty": (place.get("primaryTypeDisplayName") or {}).get("text") or "Medical Clinic",
            "facility_type": "CARDIOLOGY_CENTRE",
            "hospital": name,
            "city": location_label,
            "lat": p_lat,
            "lon": p_lon,
            "address": place.get("formattedAddress", ""),
            "contact": place.get("internationalPhoneNumber", ""),
            "emergency_phone": AMBULANCE_NUMBER,
            "experience": "",
            "rating": rating,
            "timings": opening[0] if opening else "Call to confirm OPD hours",
            "website": place.get("websiteUri", ""),
            "google_maps": maps_url,
            "profile_url": maps_url,
            "photo_initials": _initials(name),
            "distance_km": dist_km,
            "distance_text": _distance_text(dist_km),
            "source": "Google Places",
        })
    return items


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def search_doctors_near(
    location: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[int] = None,
    limit: int = 12,
    triage_level: Optional[str] = None,
) -> Dict[str, Any]:
    """Finds real cardiac/general care near a patient's own location.

    Accepts either a free-text ``location`` ("Jammu", "Pune, India") or raw
    ``lat``/``lon`` from the browser's geolocation.
    """
    key = google_api_key()
    query_label = (location or "").strip()
    resolved: Optional[Dict[str, Any]] = None

    if lat is None or lon is None:
        if not query_label:
            return {
                "status": "error", "provider": None, "doctors": [], "total_facilities": 0,
                "google_key_configured": bool(key),
                "message": "Please enter your city or area so we can find doctors near you.",
            }
        resolved = geocode_place(query_label)
        if not resolved:
            return {
                "status": "error", "provider": None, "query_location": query_label,
                "doctors": [], "total_facilities": 0,
                "google_key_configured": bool(key),
                "message": (
                    f"We could not locate '{query_label}'. Try adding the state and country, "
                    "for example 'Jammu, Jammu and Kashmir, India'."
                ),
            }
        lat, lon = resolved["lat"], resolved["lon"]

    city = (resolved or {}).get("city") or query_label or "Your area"
    location_label = resolved["display_name"] if resolved else (query_label or "your location")
    rings = (radius_km,) if radius_km else RADIUS_LADDER_KM

    doctors: List[Dict[str, Any]] = []
    used_radius = rings[0]
    provider_used = "openstreetmap"
    provider_error = ""

    # Google Places first when configured: real individual doctors, ratings
    # and review counts that free sources cannot provide.
    if key:
        try:
            found = search_google_places(key, lat, lon, rings[0], limit, location_label)
            if found:
                doctors, provider_used = _rank(found, lat, lon, rings[0] * 3, limit), "google_places"
                used_radius = rings[0]
        except Exception as exc:
            provider_error = str(exc)
            print(f"[geo_doctors] Google Places failed, using OpenStreetMap: {exc}")

    if not doctors:
        provider_used = "openstreetmap"
        collected: List[Dict[str, Any]] = []

        # Primary: Nominatim category search inside the search ring. It is fast
        # and its results carry real facility names plus their OpenStreetMap tags.
        for ring in rings:
            used_radius = ring
            collected.extend(search_nominatim(lat, lon, ring, triage_level))
            doctors = _rank(collected, lat, lon, ring, limit)
            if len(doctors) >= MIN_RESULTS_BEFORE_WIDENING:
                break

        # Secondary: Photon catches facilities whose names a category search misses.
        if len(doctors) < MIN_RESULTS_BEFORE_WIDENING:
            extra, _ = search_photon(city, lat, lon, triage_level)
            if extra:
                collected.extend(extra)
                doctors = _rank(collected, lat, lon, used_radius, limit)

        # Last resort: one tag-based Overpass sweep for small or sparse towns.
        if len(doctors) < 3:
            deep = search_overpass(lat, lon, used_radius, str(city), triage_level)
            if deep:
                collected.extend(deep)
                doctors = _rank(collected, lat, lon, used_radius, limit)

    if not doctors:
        message = (
            f"No cardiac or general medical facilities could be found near '{city}' "
            f"(searched up to {rings[-1]} km). Try a nearby larger city."
        )
        if provider_error:
            message += f" (Directory note: {provider_error})"
        return {
            "status": "error", "provider": provider_used, "query_location": query_label,
            "resolved_location": resolved, "search_radius_km": used_radius,
            "doctors": [], "total_facilities": 0,
            "google_key_configured": bool(key), "message": message,
        }

    provider_label = "Google Places (live)" if provider_used == "google_places" else "OpenStreetMap (live)"
    return {
        "status": "success",
        "provider": provider_used,
        "provider_label": provider_label,
        "query_location": query_label,
        "resolved_location": resolved,
        "resolved_city": city,
        "search_radius_km": used_radius,
        "total_facilities": len(doctors),
        "google_key_configured": bool(key),
        "doctors": doctors,
        "message": (
            f"{len(doctors)} real facilities within {used_radius} km of {city} "
            f"\u00b7 live data from {provider_label}."
        ),
    }
