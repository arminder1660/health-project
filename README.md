# CardiaNova

Chest-pain triage and cardiovascular risk screening, as a small FastAPI service with a
vanilla-JavaScript front end.

A patient describes what they feel, answers a short vitals wizard, and gets back a triage
level, a calibrated 10-year-style risk score, diet guidance, and a live list of **real**
hospitals, cardiac centres and clinics near the city they are actually in.

> **This is not a medical device and it is not a diagnosis.**
> It is a screening aid and a portfolio project. It cannot run an ECG, a troponin assay or
> a cath-lab procedure, and it does not replace a clinician. In an emergency in India dial
> **108**, or **112 / 911** for general emergency services.

---

## What it actually does

```
free text + wizard answers
        │
        ▼
[1] Rule-based clinical entity extraction      src/entity_extraction.py
        │  age, sex, BP, risk factors, pain character, radiation,
        │  triggers, relief, duration, red flags  (auditable regex, no ML)
        ▼
[2] Calibrated ML risk score                   src/train.py
        │  HistGradientBoostingClassifier wrapped in
        │  CalibratedClassifierCV(method="sigmoid", cv=5)
        ▼
[3] Four-level triage escalation               src/triage_decision.py
        │  EMERGENCY_ER        -> blood-red  (nearest emergency department)
        │  SPECIALIST_CARDIOLOGIST -> red    (cardiologist / cardiac centre)
        │  GENERAL_PHYSICIAN   -> yellow   (family clinic)
        │  HOME_CARE_LIFESTYLE -> green    (no doctor needed right now)
        ▼
[4] Diet + home remedies                       src/diet_recommender.py
        │
        └── if any clinician review is advised ──► location-aware live doctor
                                                   search (src/geo_doctors.py)
```

Two deliberate design choices worth calling out:

- **Symptom interpretation is rule-based, not a language model.** The clinical keyword
  families (crushing chest pressure, radiation to the left arm, cold sweats, exertional
  trigger, relief by rest) are small, stable and safety-critical, so explicit rules are
  preferred over a statistical model whose mistakes cannot be explained to a patient.
  `src/entity_extraction.py` is a hand-written pattern matcher and says so in its docstring.
- **Traps are handled explicitly.** "no shortness of breath" is not breathlessness, and
  "I quit smoking two years ago" is not a smoker. Both have regression tests.

## Model card

The risk model is trained on the public [UCI / Kaggle cardiovascular disease
dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
(70,000 records → 68,573 after physiological outlier removal → **68,549 after
deduplication**), with an 80/20 stratified split — 54,839 train / 13,710 held-out test rows.
Duplicates are removed *before* the split: the file ships a per-row `id`, so two records for
the same patient look distinct unless you deduplicate on the clinical columns.

| Metric (held-out test) | This model | Logistic-regression baseline |
| --- | --- | --- |
| Accuracy | **73.07%** | 72.25% |
| ROC-AUC | **79.78%** | 78.79% |
| Brier score (lower is better) | **0.1817** | 0.1879 |

Ten features: `age_years`, `gender`, `bmi`, `ap_hi`, `ap_lo`, `cholesterol`, `gluc`,
`smoke`, `alco`, `active`. All are standardised with a `StandardScaler` fitted on the
training split only.

**One split is not a measurement, so it is not published as one.** Repeated stratified
5×5 cross-validation — 25 fits, on the training split only, so the held-out rows stay
held out — gives an interval instead of a point:

| Metric | Mean (95% CI, 25 folds) |
| --- | --- |
| Accuracy | 73.60% [73.44, 73.76] |
| ROC-AUC | **80.21%** [80.04, 80.37] |
| Brier score | 0.1801 [0.1793, 0.1809] |

Read those intervals before believing any future change. A "gain" of 0.3 points of ROC-AUC
is inside the noise band, and saying so is part of reporting the result.

**Accuracy is the least interesting number here.** The outcome is 49.5 % positive in this
dataset, so a model that answers "no disease" for every single patient already scores 50.5 %.
At the threshold the app actually ships with (0.50):

| Sensitivity (real cases caught) | Specificity | PPV (alarms that are real) | NPV |
| --- | --- | --- | --- |
| 68.30% | 77.74% | 75.03% | 71.47% |

So the model alone misses about one real case in three. That is only survivable because the
model is not the only decision-maker: the rule-based triage layer escalates on red-flag
symptoms and dangerous vitals regardless of what the score says, and a high score with no
supporting clinical pattern never escalates anyone.

**Where it is weaker.** The same held-out split, broken down:

| Subgroup | n | Prevalence | ROC-AUC | Sensitivity | Specificity |
| --- | --- | --- | --- | --- | --- |
| Age under 45 | 1,971 | 30.3% | **82.39%** | 54.85% | 93.37% |
| Age 45–54 | 5,583 | 45.1% | 79.00% | 58.82% | 86.11% |
| Age 55 and over | 6,156 | 59.6% | **74.81%** | **77.00%** | 58.79% |
| Female | 8,864 | 49.2% | 79.86% | 68.16% | 78.12% |
| Male | 4,846 | 49.9% | 79.64% | 68.55% | 77.04% |

Two separate effects are visible. Ranking ability genuinely degrades with age — ROC-AUC falls
82.4 → 74.8 — and that is not an artefact of the base rate, because AUC is prevalence-independent.
The sensitivity and specificity columns moving in opposite directions *are* largely a base-rate
effect: as prevalence climbs from 30 % to 60 %, a fixed 0.50 threshold necessarily catches more
cases and raises more false alarms in the older band. The practical consequence is that the
model's ranking is weakest in the band where most heart disease actually occurs, so the triage
rules have to carry more of the weight there.

Sex shows no meaningful gap (79.86 % vs 79.64 % AUC, 68.16 % vs 68.55 % sensitivity), which is
worth stating explicitly because it is a question every reviewer asks.

**What the model is actually using.** Permutation importance (drop in ROC-AUC):

| Feature | Importance |
| --- | --- |
| Systolic BP | **+0.175** |
| Age | +0.031 |
| Cholesterol | +0.030 |
| BMI | +0.005 |
| Diastolic BP | +0.002 |
| Physical activity | +0.002 |
| Smoking | +0.001 |
| Everything else (glucose, alcohol, sex) | ≤ +0.001 each |

The honest reading: this model is close to "systolic blood pressure, age and cholesterol, plus a
little BMI". Historical systolic BP alone accounts for most of the ranking ability it has. That
is a property of the *dataset*, not of the algorithm, and it is why the ROC-AUC sits stubbornly
near 0.80 however the model is tuned: these ten columns do not contain much more signal to
extract.

`pulse_pressure` (systolic − diastolic) was therefore removed. It is a deterministic function of
two columns already in the set, and it measured at **+0.0001** importance. Dropping it moved the
held-out numbers by less than the cross-validation interval — 73.15 % → 73.07 % accuracy,
79.80 % → 79.78 % ROC-AUC — which is exactly what removing a dead feature should do.

`alco` (+0.0006) and `active` (+0.002) are nearly as thin, but they stay, because `/api/simulate`
lets a patient change precisely those two answers and see the effect. A knob the product offers
has to be wired to the model, even when its average effect is small.

Self-reported lab values are accepted as unknown (`null`) rather than being guessed: a patient
who does not know their cholesterol gets a score from what they *did* report, and no feature is
ever inferred from an unrelated condition.

The honest caveats, the same strings the app itself publishes:

- Trained on a single public dataset; not validated on local patient populations.
- Symptom interpretation is rule-based (auditable regex), not a language model.
- Output is a screening aid only and is not a medical diagnosis.

`GET /api/model-info` returns all of the above — the held-out metrics, the cross-validated
intervals, the operating-point confusion matrix, the subgroup table, permutation importance,
60-point ROC and calibration curves, and the caveats. The home page renders them in a "Model
Transparency" panel, so the numbers are auditable rather than a badge on a slide. If the
artifact has not been built, the endpoint returns **503** with an instruction instead of quietly
serving a guessed number.

Retrain at any time:

```bash
python -m src.train
```

## Live, location-aware doctor referral

When the triage level is anything other than safe home care, the app asks for the patient's
city or area, geocodes it, and then searches around those coordinates. **Results are never
fabricated** — every entry carries a `source` and a `profile_url` so it can be checked.

| Order | Provider | Key needed | What it returns |
| --- | --- | --- | --- |
| 1 | Nominatim (OpenStreetMap) | no | real facilities bounded to a search ring |
| 2 | Photon | no | facilities a category search misses |
| 3 | Overpass API | no | deeper sweep for small or sparse towns |
| — | Google Places API (New) | optional | preferred when configured: named consultants, star ratings, review counts |

- Search rings widen automatically: **15 km → 40 km → 80 km**, stopping once
  `MIN_RESULTS_BEFORE_WIDENING = 6` facilities are found, so a metro resolves on the first
  ring while a small town expands. The radius actually used is reported back to the UI.
- Results are ranked by distance-tempered relevance, and the ranking is tuned by triage
  level (an `EMERGENCY_ER` verdict prefers hospitals over clinics).
- Nominatim's usage policy is respected: one request per second, a descriptive
  `User-Agent`, and bounded viewbox queries.
- **Safety filter.** Alternative-medicine providers (ayurveda, homeopathy, unani, siddha,
  naturopathy, acupuncture, chiropractic) are rejected outright, as are eye/dental/ortho/
  derma/psych/physio/veterinary places and stand-alone diagnostic labs. A place that cannot
  investigate a heart is not a helpful referral for suspected cardiac disease. A broad
  multi-speciality hospital keeps its place even if a unit in its name matches the filter.

To enable Google Places, copy `.env.example` to `.env` and set `GOOGLE_PLACES_API_KEY`.
With no key the app runs entirely on the free OpenStreetMap stack. `.env` is git-ignored.
`GET /api/doctor-search-status` reports which provider is currently active.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/triage` | full pipeline: entities → risk → triage → diet → referral hints |
| `POST` | `/api/simulate` | what-if risk reduction between a baseline and a target profile |
| `GET` | `/api/nearby-doctors` | live doctor search by `location`, or by `lat`/`lon` |
| `GET` | `/api/model-info` | metrics, cross-validated intervals, operating point, subgroups, importance, ROC + calibration curves, caveats |
| `GET` | `/api/doctor-search-status` | which live directory is active |
| `GET` | `/` | the web app; `/docs` for interactive OpenAPI docs |

`/api/triage` accepts either free text (`query_text`) or structured wizard fields, or both.
Free text is parsed first and the wizard values then override or supplement it. Self-reported
lab values (`cholesterol`, `gluc`) and lifestyle answers (`alco`, `active`) are optional and
are passed to the model verbatim; `null` means "not known" and is scored as a genuine missing
value rather than being replaced by a default or inferred from an unrelated condition.

## Quickstart

Windows, one step (creates nothing, trains the model once if missing, opens a browser):

```bat
run.bat
```

Any platform, manually:

```bash
python -m venv venv
venv/Scripts/activate        # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python -m src.train          # one-time: writes models/cardio_model.joblib
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>. Requires Python 3.12.

Docker:

```bash
docker build -t cardianova .
docker run -p 8000:8000 --env-file .env cardianova
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

128 tests, no network access — the HTTP layer is monkeypatched. They cover the extractor's
clinical patterns and their negations, the triage safety contract (emergency escalation,
`theme` ↔ `triage_level` consistency, high model risk alone must *not* escalate), the
non-cardiac provider filter, distance/relevance ranking, the model card's internal consistency
(confusion counts must agree with the reported sensitivity, the sex subgroups must partition
the test set), and every API endpoint through `fastapi.testclient`.

CI (`.github/workflows/ci.yml`) runs on every push: install, retrain the model to prove the
training pipeline still works, run the suite, then boot the server and smoke-test
`/api/model-info` and `/`.

## Project layout

```
server.py                  FastAPI app, request models, endpoints
src/entity_extraction.py   rule-based clinical entity extraction
src/train.py               training, calibration, diagnostics, inference
src/triage_decision.py     four-level escalation engine
src/diet_recommender.py    diet and home-remedy guidance
src/geo_doctors.py         live doctor discovery by patient location
web/                       index.html, app.js, style.css
data/cardio_train.csv      committed so training works offline
models/cardio_model.joblib committed so a fresh clone runs immediately
tests/                     pytest suite
```

## Data and privacy

The app has no database. Nothing the patient types is stored, logged or sent anywhere except
the live geocoding/POI lookups needed to answer "who is near me". The only third-party
traffic is to the public OpenStreetMap services listed above (or Google Places if you
configure a key).

## License

MIT — see [LICENSE](LICENSE). The licence includes an explicit non-medical-device notice.
