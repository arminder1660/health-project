/**
 * CardiaNova — Client Application Logic (Redesigned)
 */

const state = {
  currentStage: 'home',
  currentStep: 1,
  selectedProblem: null,
  formData: {
    age: 52, gender: 'MALE', height: 170, weight: 74,
    sensation: 'HEAVY_PRESSURE_TIGHTNESS', radiation: 'NONE',
    trigger: 'PHYSICAL_EXERTION', duration: 'MINUTES_5_TO_15',
    systolic_bp: 142, diastolic_bp: 90,
    has_hypertension: true, has_diabetes: false, is_smoker: false,
    // null means "not answered". The server forwards null as a genuine missing
    // value so the risk model marginalises over it, rather than the app
    // inventing a "normal" reading the patient never reported.
    cholesterol: null, gluc: null, alco: null, active: null,
    red_flags: [], query_text: '',
  },
  breathingInterval: null,
  isBreathing: false,
  triageLevel: null,      // last triage level, used to tune the doctor search
  lastLocation: '',       // most recent city/area the user searched for
};

const PROBLEM_LABELS = {
  angina:       'Chest Heaviness / Pressure',
  palpitations: 'Heart Flutters & Racing Pulse',
  hypertension: 'High Blood Pressure',
  reflux:       'Heartburn vs. Heart Attack',
};

/* ---- Output sanitisation ----
   Every value rendered into the page comes from an API response, and some of
   those values originate from live third-party directories. Nothing is ever
   interpolated into innerHTML without going through escapeHtml first, so a
   facility name such as `<img src=x onerror=...>` stays inert text. */
function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* Only allow links that cannot execute script (http, https, tel, mailto). */
function safeUrl(value) {
  const raw = String(value == null ? '' : value).trim();
  if (!raw) return '';
  if (/^(https?:|tel:|mailto:)/i.test(raw)) return raw;
  return '';
}

/* ---- Dynamic Risk Theme System ---- */
const THEME_CONFIG = {
  'green': {
    badgeText: '🟢 Safe (No Doctor)',
    emergencyText: 'Safe: Home Care & Lifestyle Monitoring Active',
    brandEmoji: '💚',
  },
  'yellow': {
    badgeText: '🟡 Moderate Risk (GP Review)',
    emergencyText: 'Moderate Risk: Primary Care Review in 1–2 Weeks',
    brandEmoji: '💛',
  },
  'red': {
    badgeText: '🔴 In Risk (Cardiologist)',
    emergencyText: 'In Risk: Consult Specialist Cardiologist within 48–72h',
    brandEmoji: '❤️',
  },
  'blood-red': {
    badgeText: '🩸 High Extreme Risk',
    emergencyText: 'HIGH EXTREME RISK: Hospital Emergency (ER) Immediately',
    brandEmoji: '🚨',
  },
};

function setTheme(themeName, persist = true) {
  if (!['green', 'yellow', 'red', 'blood-red'].includes(themeName)) {
    resetTheme();
    return;
  }
  document.documentElement.setAttribute('data-theme', themeName);
  document.body.className = 'theme-' + themeName;

  const cfg = THEME_CONFIG[themeName];
  const badgeText = document.getElementById('theme-status-text');
  if (badgeText && cfg) badgeText.textContent = cfg.badgeText;

  const emerText = document.getElementById('emergency-tag-text');
  if (emerText && cfg) emerText.textContent = cfg.emergencyText;

  const brandIcon = document.getElementById('main-brand-icon');
  if (brandIcon && cfg) brandIcon.innerHTML = cfg.brandEmoji;

  // Highlight active theme button
  document.querySelectorAll('.t-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById('t-btn-' + themeName);
  if (activeBtn) activeBtn.classList.add('active');

  if (persist) {
    try { localStorage.setItem('cardianova_theme', themeName); } catch (e) {}
  }
}

function resetTheme() {
  document.documentElement.removeAttribute('data-theme');
  document.body.className = '';
  const badgeText = document.getElementById('theme-status-text');
  if (badgeText) badgeText.textContent = 'Theme: Default';

  const emerText = document.getElementById('emergency-tag-text');
  if (emerText) emerText.textContent = 'Emergency: 911 · 112 · 108';

  const brandIcon = document.getElementById('main-brand-icon');
  if (brandIcon) brandIcon.innerHTML = '❤️';

  document.querySelectorAll('.t-btn').forEach(btn => btn.classList.remove('active'));
  try { localStorage.removeItem('cardianova_theme'); } catch (e) {}
}

/* ---- Stage navigation ---- */
function switchStage(name) {
  document.querySelectorAll('.stage').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

  const el = document.getElementById('stage-' + name);
  if (el) { el.classList.add('active'); state.currentStage = name; window.scrollTo({ top: 0, behavior: 'smooth' }); }

  const navMap = { home:'nav-home', wizard:'nav-consult', report:'nav-consult', simulator:'nav-sim', doctors:'nav-docs' };
  const btn = document.getElementById(navMap[name]);
  if (btn) btn.classList.add('active');

  if (name === 'doctors') {
    // Reuse the city the patient already typed during the assessment, if any.
    const input = document.getElementById('full-loc-input');
    if (input && !input.value && state.lastLocation) input.value = state.lastLocation;
  }
  if (name === 'simulator') updateSimulation();
}

/* ---- Problem selector ---- */
function selectProblem(type) {
  state.selectedProblem = type;

  // Apply preset values
  if (type === 'angina')       { state.formData.sensation = 'HEAVY_PRESSURE_TIGHTNESS'; state.formData.trigger = 'PHYSICAL_EXERTION'; state.formData.duration = 'MINUTES_5_TO_15'; state.formData.systolic_bp = 145; }
  else if (type === 'palpitations') { state.formData.sensation = 'PALPITATIONS_ARRHYTHMIA'; state.formData.trigger = 'EMOTIONAL_STRESS'; }
  else if (type === 'hypertension') { state.formData.has_hypertension = true; state.formData.systolic_bp = 152; state.formData.diastolic_bp = 96; state.formData.sensation = 'HEAVY_PRESSURE_TIGHTNESS'; }
  else if (type === 'reflux')       { state.formData.sensation = 'BURNING_ACID_REFLUX'; state.formData.trigger = 'POST_PRANDIAL_MEALS'; }

  // Show badge in sidebar
  const badge = document.getElementById('wiz-problem-badge');
  const label = document.getElementById('wiz-problem-label');
  if (badge && label) { badge.style.display = 'block'; label.textContent = PROBLEM_LABELS[type] || type; }

  state.currentStep = 1;
  switchStage('wizard');
  updateWizardUI();
}

/* ---- Wizard UI ---- */
function updateWizardUI() {
  const total = 4;
  for (let i = 1; i <= total; i++) {
    const panel     = document.getElementById('wizard-step-' + i);
    const circle    = document.getElementById('sc-' + i);
    const sideItem  = document.getElementById('wiz-sidebar-' + i);

    if (panel) panel.classList.toggle('active', i === state.currentStep);

    if (circle) {
      circle.classList.remove('active', 'done');
      if (i === state.currentStep) circle.classList.add('active');
      else if (i < state.currentStep) circle.classList.add('done');
    }
    if (sideItem) {
      sideItem.classList.remove('active', 'done');
      if (i === state.currentStep) sideItem.classList.add('active');
      else if (i < state.currentStep) sideItem.classList.add('done');
    }
  }

  const fill = document.getElementById('wiz-progress-fill');
  if (fill) fill.style.width = (state.currentStep / total * 100) + '%';

  const prevBtn   = document.getElementById('wiz-prev-btn');
  const nextBtn   = document.getElementById('wiz-next-btn');
  const submitBtn = document.getElementById('wiz-submit-btn');
  if (prevBtn)   prevBtn.style.display   = state.currentStep > 1 ? 'inline-flex' : 'none';
  if (nextBtn)   nextBtn.style.display   = state.currentStep < total ? 'inline-flex' : 'none';
  if (submitBtn) submitBtn.style.display = state.currentStep === total ? 'inline-flex' : 'none';
}

function nextStep() { if (state.currentStep < 4) { state.currentStep++; updateWizardUI(); } }
function prevStep() { if (state.currentStep > 1) { state.currentStep--; updateWizardUI(); } }

function setChip(key, val, el) {
  state.formData[key] = val;
  el.parentElement.querySelectorAll('.chip').forEach(b => b.classList.remove('selected'));
  el.classList.add('selected');
}

function setBuild(ht, wt, el) {
  state.formData.height = ht; state.formData.weight = wt;
  el.parentElement.querySelectorAll('.chip').forEach(b => b.classList.remove('selected'));
  el.classList.add('selected');
}

function toggleMultiChip(el, type) {
  el.classList.toggle('selected');
  const on = el.classList.contains('selected');
  if (type === 'htn')   state.formData.has_hypertension = on;
  if (type === 'diab')  state.formData.has_diabetes = on;
  if (type === 'smk')   state.formData.is_smoker = on;
  if (type === 'sweat') {
    if (on) { if (!state.formData.red_flags.includes('COLD_SWEATING_DIAPHORESIS')) state.formData.red_flags.push('COLD_SWEATING_DIAPHORESIS'); }
    else     state.formData.red_flags = state.formData.red_flags.filter(x => x !== 'COLD_SWEATING_DIAPHORESIS');
  }
}

/* ---- Submit ---- */
async function submitAssessment() {
  const ageEl = document.getElementById('wiz-age');
  const sysEl = document.getElementById('wiz-sys');
  const diaEl = document.getElementById('wiz-dia');
  const txtEl = document.getElementById('wiz-text');
  if (ageEl?.value) state.formData.age = parseInt(ageEl.value, 10);
  if (sysEl?.value) state.formData.systolic_bp = parseInt(sysEl.value, 10);
  if (diaEl?.value) state.formData.diastolic_bp = parseInt(diaEl.value, 10);
  if (txtEl)        state.formData.query_text = txtEl.value;

  const btn = document.getElementById('wiz-submit-btn');
  if (btn) { btn.innerHTML = '<span class="spinner"></span> Analysing...'; btn.disabled = true; }

  try {
    const resp = await fetch('/api/triage', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state.formData),
    });
    const data = await resp.json();
    if (!resp.ok) {
      alert(data.detail || 'The screening service could not complete this assessment. Please try again.');
      return;
    }
    renderReport(data);
    switchStage('report');
  } catch (err) {
    console.error(err);
    alert('Unable to connect to the clinical engine. Please ensure the server is running.');
  } finally {
    if (btn) { btn.innerHTML = 'Generate My Care Plan \uD83E\uDEBA'; btn.disabled = false; }
  }
}

/* ---- Render Report ---- */
function renderReport(data) {
  const triage = data.triage;
  const diet   = data.diet;

  // Apply theme according to user requirements:
  // "evreytime whn person didnt need to consult doctor the theme should turn to green and when theres a moderate risk it should change to yellow in risk it should change to red in high extreme risk blood red"
  let theme = 'green';
  if (triage.theme) {
    theme = triage.theme;
  } else if (triage.triage_level === 'EMERGENCY_ER') {
    theme = 'blood-red';
  } else if (triage.triage_level === 'SPECIALIST_CARDIOLOGIST') {
    theme = 'red';
  } else if (triage.triage_level === 'GENERAL_PHYSICIAN') {
    theme = 'yellow';
  } else {
    theme = 'green'; // HOME_CARE_LIFESTYLE / Person didn't need to consult doctor
  }

  // Turn entire UI theme to matching color
  setTheme(theme, true);

  // Verdict banner
  const banner = document.getElementById('report-verdict-banner');
  if (banner) {
    banner.className = 'verdict-banner ' + theme;
    document.getElementById('report-urgency').textContent     = 'Clinical Triage · Urgency: ' + triage.urgency;
    document.getElementById('report-headline').textContent    = triage.verdict_headline;
    document.getElementById('report-doctor-opinion').textContent = triage.consult_big_doctor;
  }

  // Reasoning
  document.getElementById('report-suspicion').textContent       = triage.primary_suspicion;
  document.getElementById('report-reasoning-text').textContent  = triage.reasoning;
  const rfList = document.getElementById('report-red-flags');
  if (rfList) rfList.innerHTML = (triage.red_flag_warnings || []).map(w => '<li>' + escapeHtml(w) + '</li>').join('');

  // Diet
  document.getElementById('report-diet-title').textContent = diet.protocol_name;
  document.getElementById('report-diet-goal').textContent  = diet.goal;

  const eatEl = document.getElementById('report-foods-eat');
  if (eatEl) eatEl.innerHTML = diet.foods_to_eat.map(f => `
    <div class="food-pill good">
      <span class="food-pill-icon">\u2705</span>
      <div>
        <div class="food-pill-name">${escapeHtml(f.food)}</div>
        <div class="food-pill-desc">${escapeHtml(f.reason)}</div>
      </div>
    </div>`).join('');

  const avoidEl = document.getElementById('report-foods-avoid');
  if (avoidEl) avoidEl.innerHTML = diet.foods_to_avoid.map(f => `
    <div class="food-pill avoid">
      <span class="food-pill-icon">\u274C</span>
      <div>
        <div class="food-pill-name">${escapeHtml(f.food)}</div>
        <div class="food-pill-desc">${escapeHtml(f.reason)}</div>
      </div>
    </div>`).join('');

  const menuEl = document.getElementById('report-menu-grid');
  if (menuEl) menuEl.innerHTML = `
    <div class="meal-item"><div class="meal-label">Breakfast</div><div class="meal-text">${escapeHtml(diet.sample_menu.Breakfast)}</div></div>
    <div class="meal-item"><div class="meal-label">Lunch</div><div class="meal-text">${escapeHtml(diet.sample_menu.Lunch)}</div></div>
    <div class="meal-item"><div class="meal-label">Evening Snack</div><div class="meal-text">${escapeHtml(diet.sample_menu['Evening Snack'])}</div></div>
    <div class="meal-item"><div class="meal-label">Dinner</div><div class="meal-text">${escapeHtml(diet.sample_menu.Dinner)}</div></div>`;

  // Remedies
  const remEl = document.getElementById('report-remedies-list');
  if (remEl) remEl.innerHTML = diet.remedies.map(r => `
    <div class="remedy-card">
      <div class="remedy-icon">\uD83C\uDF3F</div>
      <div>
        <div class="remedy-name">${escapeHtml(r.practice)}</div>
        <div class="remedy-how">${escapeHtml(r.how_to_do)}</div>
        <div class="remedy-benefit">\u2713 ${escapeHtml(r.clinical_benefit)}</div>
      </div>
    </div>`).join('');

  loadReportDoctorsPanel(data);
}

/* ---- Report tabs ---- */
function switchReportTab(key) {
  document.querySelectorAll('.rtab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.report-panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById('rtab-' + key);
  if (panel) panel.classList.add('active');
  const btn = Array.from(document.querySelectorAll('.rtab')).find(b => b.getAttribute('onclick')?.includes(key));
  if (btn) btn.classList.add('active');
}

/* ---- Breathing coach ---- */
function toggleBreathing() {
  const btn    = document.getElementById('btn-breathing');
  const circle = document.getElementById('breath-circle');
  if (state.isBreathing) {
    clearInterval(state.breathingInterval); state.isBreathing = false;
    btn.innerHTML = 'Start 1-Minute Session';
    circle.textContent = 'Ready'; circle.style.transform = 'scale(1)';
    return;
  }
  state.isBreathing = true;
  btn.innerHTML = 'Stop Session';
  let phase = 0, count = 4;
  circle.textContent = 'Inhale (' + count + ')'; circle.style.transform = 'scale(1.35)';
  state.breathingInterval = setInterval(() => {
    count--;
    if (count <= 0) {
      phase = (phase + 1) % 3;
      if (phase === 0)      { count = 4; circle.textContent = 'Inhale (' + count + ')'; circle.style.transform = 'scale(1.35)'; }
      else if (phase === 1) { count = 7; circle.textContent = 'Hold (' + count + ')';   circle.style.transform = 'scale(1.35)'; }
      else                  { count = 8; circle.textContent = 'Exhale (' + count + ')'; circle.style.transform = 'scale(1)'; }
    } else {
      if (phase === 0) circle.textContent = 'Inhale (' + count + ')';
      if (phase === 1) circle.textContent = 'Hold (' + count + ')';
      if (phase === 2) circle.textContent = 'Exhale (' + count + ')';
    }
  }, 1000);
}

/* ---- Doctors: live, location-aware lookup ----
   When the triage engine advises seeing a clinician we ask the patient where
   they are, then fetch REAL facilities near that place only (OpenStreetMap by
   default, Google Places when a key is configured on the server). */
const DOCTOR_UI = {
  report: { input: 'report-loc-input', status: 'report-loc-status', grid: 'report-doctors-list', panel: 'report-loc-panel' },
  full:   { input: 'full-loc-input',   status: 'full-loc-status',   grid: 'full-doctors-grid',   panel: 'full-loc-panel'   },
};

function loadReportDoctorsPanel(data) {
  const triage = data.triage || {};
  state.triageLevel = triage.triage_level || null;

  const grid  = document.getElementById(DOCTOR_UI.report.grid);
  const panel = document.getElementById(DOCTOR_UI.report.panel);
  const needsDoctor = data.consultation_needed !== false;

  // Safe / home-care verdicts never need a location prompt.
  if (!needsDoctor) {
    if (panel) panel.style.display = 'none';
    if (grid) grid.innerHTML = `
      <div class="doc-note ok">
        Your assessment did not require a doctor visit. If your symptoms change,
        consult a clinician &mdash; then come back here and we will find one near you.
      </div>`;
    return;
  }

  if (panel) panel.style.display = '';
  if (grid) grid.innerHTML = '';

  const specialist = (data.doctor_search && data.doctor_search.recommended_specialist) || '';
  setDoctorStatus('report', specialist
    ? `Recommended: ${specialist}. Enter your city or area to see real options near you.`
    : 'Enter your city or area to see real options near you.', 'info');
}

function searchDoctorsFrom(scope) {
  const ui = DOCTOR_UI[scope];
  if (!ui) return;
  const input = document.getElementById(ui.input);
  const place = (input && input.value || '').trim();
  if (!place) {
    setDoctorStatus(scope, 'Please enter your city or area first.', 'warn');
    return;
  }
  state.lastLocation = place;
  fetchNearbyDoctors(scope, { location: place });
}

function useMyLocation(scope) {
  const ui = DOCTOR_UI[scope];
  if (!ui) return;
  if (!navigator.geolocation) {
    setDoctorStatus(scope, 'Your browser cannot share a location. Please type your city instead.', 'warn');
    return;
  }
  setDoctorStatus(scope, 'Getting your location\u2026', 'info');
  navigator.geolocation.getCurrentPosition(
    pos => fetchNearbyDoctors(scope, { lat: pos.coords.latitude, lon: pos.coords.longitude }),
    () => setDoctorStatus(scope, 'Location permission was denied. Please type your city instead.', 'warn'),
    { timeout: 10000, enableHighAccuracy: false }
  );
}

function setDoctorStatus(scope, message, kind) {
  const el = document.getElementById(DOCTOR_UI[scope].status);
  if (!el) return;
  el.textContent = message || '';
  el.className = 'loc-status' + (kind ? ' ' + kind : '');
}

async function fetchNearbyDoctors(scope, params) {
  const ui   = DOCTOR_UI[scope];
  const grid = document.getElementById(ui.grid);

  const qs = new URLSearchParams();
  if (params.location) qs.set('location', params.location);
  if (params.lat != null) qs.set('lat', params.lat);
  if (params.lon != null) qs.set('lon', params.lon);
  if (state.triageLevel) qs.set('triage_level', state.triageLevel);

  if (grid) grid.innerHTML = '<div class="doc-note loading"><span class="spinner"></span> Searching live maps for doctors near you\u2026</div>';
  setDoctorStatus(scope, 'Looking up real facilities on live maps\u2026', 'info');

  try {
    const res  = await fetch('/api/nearby-doctors?' + qs.toString());
    const data = await res.json();

    if (data.status !== 'success') {
      if (grid) grid.innerHTML = '';
      setDoctorStatus(scope, data.message || 'No doctors were found. Try a nearby larger city.', 'warn');
      return;
    }

    if (grid) grid.innerHTML = data.doctors.map(doctorCard).join('');
    setDoctorStatus(scope, data.message, 'ok');
  } catch (err) {
    console.error('Nearby doctors error:', err);
    if (grid) grid.innerHTML = '';
    setDoctorStatus(scope, 'Could not reach the live directory. Check your connection and try again.', 'warn');
  }
}

function doctorCard(d) {
  const meta = [d.distance_text, d.address].filter(Boolean).join(' \u00B7 ');
  const tel    = safeUrl(d.contact ? 'tel:' + String(d.contact).replace(/[^\d+]/g, '') : '');
  const web    = safeUrl(d.website);
  const prof   = safeUrl(d.profile_url);
  const maps   = safeUrl(d.google_maps);
  const links = [
    tel   ? `<a class="doc-link" href="${escapeHtml(tel)}">\u260E ${escapeHtml(d.contact)}</a>` : '',
    web   ? `<a class="doc-link" href="${escapeHtml(web)}" target="_blank" rel="noopener noreferrer">Website</a>` : '',
    prof  ? `<a class="doc-link" href="${escapeHtml(prof)}" target="_blank" rel="noopener noreferrer">Verify on ${escapeHtml(d.source || 'source')}</a>` : '',
    maps  ? `<a class="doc-link" href="${escapeHtml(maps)}" target="_blank" rel="noopener noreferrer">Directions</a>` : '',
  ].filter(Boolean).join('');

  return `
    <div class="doctor-card">
      <div class="doctor-avatar">${escapeHtml(d.photo_initials || 'DR')}</div>
      <div style="flex:1; min-width:0;">
        <div class="doctor-name">${escapeHtml(d.name)}</div>
        <div class="doctor-spec">${escapeHtml(d.specialty)}${d.rating ? ' \u00B7 ' + escapeHtml(d.rating) : ''}</div>
        <div class="doctor-hospital">\uD83C\uDFE5 ${escapeHtml(meta || d.city || '')}</div>
        <div class="doctor-links">${links}</div>
      </div>
      <span class="doc-source" title="Live data source">${escapeHtml(d.source || 'Live')}</span>
    </div>`;
}

/* ---- Simulator ---- */
async function updateSimulation() {
  const baseSys   = parseInt(document.getElementById('sim-base-sys')?.value   || 155, 10);
  const baseWt    = parseFloat(document.getElementById('sim-base-wt')?.value  || 86);
  const targetSys = parseInt(document.getElementById('sim-target-sys')?.value || 122, 10);
  const targetWt  = parseFloat(document.getElementById('sim-target-wt')?.value|| 76);

  document.getElementById('val-base-sys').textContent   = baseSys   + ' mm Hg';
  document.getElementById('val-base-wt').textContent    = baseWt    + ' kg';
  document.getElementById('val-target-sys').textContent = targetSys + ' mm Hg';
  document.getElementById('val-target-wt').textContent  = targetWt  + ' kg';

  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_age:54, base_sys:baseSys, base_dia:95, base_wt:baseWt, base_ht:172, base_smk:true, base_active:false, target_sys:targetSys, target_dia:80, target_wt:targetWt, target_smk:false, target_active:true }),
    });
    const sim = await res.json();
    document.getElementById('sim-base-prob').textContent   = sim.base_risk.probability + '%';
    document.getElementById('sim-base-cat').textContent    = sim.base_risk.risk_category;
    document.getElementById('sim-target-prob').textContent = sim.target_risk.probability + '%';
    document.getElementById('sim-target-cat').textContent  = sim.target_risk.risk_category;
    document.getElementById('sim-diff-text').textContent   = '\u2212' + sim.risk_reduction + '%';

    // If currently on simulator stage, align theme to projected target risk
    if (state.currentStage === 'simulator' && sim.target_risk?.theme) {
      setTheme(sim.target_risk.theme, false);
    }
  } catch (err) { console.error('Simulation error:', err); }
}

/* ---- Model transparency card ----
   Fetches the real held-out metrics, the baseline the model is compared
   against, permutation feature importance and the ROC / calibration curves,
   and renders them so the numbers can be inspected instead of trusted. */
function metricCard(label, value, hint) {
  return `
    <div class="metric-card">
      <div class="metric-value">${escapeHtml(value)}</div>
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-hint">${escapeHtml(hint)}</div>
    </div>`;
}

function lineChart(points, xLabel, yLabel, diagonal) {
  if (!points || !points.length) return '';
  const w = 260, h = 200, pad = 26;
  const sx = v => pad + v * (w - pad * 2);
  const sy = v => h - pad - v * (h - pad * 2);
  const path = points.map(p => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
  const diag = diagonal
    ? `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="#cbd5e1" stroke-dasharray="4 4" stroke-width="1.5"/>`
    : '';
  return `
    <figure class="curve-figure">
      <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${escapeHtml(yLabel)} versus ${escapeHtml(xLabel)}" class="curve-svg">
        <rect x="${pad}" y="${pad}" width="${w - pad * 2}" height="${h - pad * 2}" fill="#f8fafc" stroke="#e2e8f0"/>
        ${diag}
        <polyline points="${path}" fill="none" stroke="#dc2626" stroke-width="2.5" stroke-linejoin="round"/>
        <text x="${w / 2}" y="${h - 6}" text-anchor="middle" class="curve-axis">${escapeHtml(xLabel)}</text>
        <text x="10" y="${h / 2}" text-anchor="middle" class="curve-axis" transform="rotate(-90 10 ${h / 2})">${escapeHtml(yLabel)}</text>
      </svg>
    </figure>`;
}

async function loadModelInfo() {
  const body = document.getElementById('model-info-body');
  if (!body) return;
  body.innerHTML = '<div class="doc-note loading"><span class="spinner"></span> Loading the model card\u2026</div>';

  try {
    const resp = await fetch('/api/model-info');
    const info = await resp.json();
    if (!resp.ok) {
      body.innerHTML = `<div class="doc-note warn">${escapeHtml(info.detail || 'The model card is unavailable.')}</div>`;
      return;
    }

    const m = info.metrics || {};
    const b = info.baseline_metrics || {};
    const top = (info.feature_importance || []).slice(0, 6);
    const maxImp = top.length ? Math.max(...top.map(f => Math.abs(f.importance))) : 1;

    const bars = top.map(f => `
      <div class="imp-row">
        <div class="imp-label">${escapeHtml(f.label)}</div>
        <div class="imp-track"><div class="imp-fill" style="width:${Math.max(2, Math.abs(f.importance) / maxImp * 100).toFixed(1)}%"></div></div>
        <div class="imp-value">${f.importance.toFixed(3)}</div>
      </div>`).join('');

    const caveats = (info.caveats || []).map(c => `<li>${escapeHtml(c)}</li>`).join('');

    body.innerHTML = `
      <div class="metric-grid">
        ${metricCard('Accuracy', (m.accuracy ?? '\u2014') + '%', `on ${Number(m.test_samples || 0).toLocaleString()} held-out patients`)}
        ${metricCard('ROC-AUC', (m.roc_auc ?? '\u2014') + '%', 'ability to rank risk correctly')}
        ${metricCard('Brier score', (m.brier_score ?? '\u2014'), 'probability calibration, lower is better')}
        ${metricCard('Trained on', Number(info.dataset?.rows_used || 0).toLocaleString(), 'cleaned patient records')}
      </div>

      <div class="info-card model-baseline">
        <div class="info-card-title">Honest baseline comparison</div>
        <table class="baseline-table">
          <thead><tr><th>Model</th><th>Accuracy</th><th>ROC-AUC</th><th>Brier</th></tr></thead>
          <tbody>
            <tr><td>Gradient boosting (calibrated)</td><td>${escapeHtml(m.accuracy ?? '\u2014')}%</td><td>${escapeHtml(m.roc_auc ?? '\u2014')}%</td><td>${escapeHtml(m.brier_score ?? '\u2014')}</td></tr>
            <tr><td>${escapeHtml(b.name || 'Logistic regression baseline')}</td><td>${escapeHtml(b.accuracy ?? '\u2014')}%</td><td>${escapeHtml(b.roc_auc ?? '\u2014')}%</td><td>${escapeHtml(b.brier_score ?? '\u2014')}</td></tr>
          </tbody>
        </table>
        <p class="model-caveat-note">The gap over a plain logistic regression is small. That is the honest result, and it is why this tool is presented as a screening aid rather than a diagnostic breakthrough.</p>
      </div>

      <div class="model-split">
        <div class="info-card">
          <div class="info-card-title">What actually drives the score</div>
          <p class="model-hint">Permutation importance on held-out data &mdash; how much ROC-AUC is lost when each input is shuffled.</p>
          <div class="imp-list">${bars}</div>
        </div>
        <div class="info-card">
          <div class="info-card-title">Reliability diagrams</div>
          <div class="curve-pair">
            ${lineChart(info.roc_curve, 'False positive rate', 'True positive rate', true)}
            ${lineChart(info.calibration_curve, 'Predicted risk', 'Observed risk', true)}
          </div>
          <p class="model-hint">Left: ROC (the dashed line is random guessing). Right: calibration &mdash; points on the dashed line mean a stated 30% risk really happens 30% of the time.</p>
        </div>
      </div>

      <div class="info-card">
        <div class="info-card-title">Limitations this project will not hide</div>
        <ul class="model-caveats">${caveats}</ul>
      </div>`;
  } catch (err) {
    console.error('Model info error:', err);
    body.innerHTML = '<div class="doc-note warn">Could not load the model card. Check that the server is running and the model has been trained.</div>';
  }
}

/* ---- Boot ---- */
window.addEventListener('DOMContentLoaded', () => {
  updateWizardUI();
  try {
    const savedTheme = localStorage.getItem('cardianova_theme');
    if (savedTheme) setTheme(savedTheme, false);
  } catch (e) {}
  loadModelInfo();
});
