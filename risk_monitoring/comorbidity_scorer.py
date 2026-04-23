"""
Co-morbidity risk scoring for CKD patients.
Simultaneous scoring for:
  - Diabetes progression risk
  - Hypertension cardiovascular risk
  - Cardiovascular disease (Framingham-based) risk
"""
import math


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def diabetes_risk_score(patient: dict) -> dict:
    """
    Estimates diabetes complication risk score (0-100).
    Uses HbA1c proxy (blood glucose), duration, and organ involvement.
    """
    score = 0
    details = []

    bg = float(patient.get('bgr') or patient.get('blood_glucose') or 120)
    if bg > 200:
        score += 30
        details.append(f"Severely elevated glucose ({bg:.0f} mg/dL)")
    elif bg > 140:
        score += 15
        details.append(f"Elevated glucose ({bg:.0f} mg/dL)")

    age = float(patient.get('age', 55))
    if age > 65:
        score += 10
    elif age > 50:
        score += 5

    htn = str(patient.get('htn', 'no')).lower() in ('yes', '1', 'true')
    if htn:
        score += 15
        details.append("Hypertension (major DM complication multiplier)")

    albumin = float(patient.get('al') or patient.get('albumin') or 0)
    uacr = float(patient.get('uacr') or 0)
    if albumin >= 3 or uacr > 300:
        score += 20
        details.append("Significant proteinuria — diabetic nephropathy likely")
    elif albumin >= 1 or uacr > 30:
        score += 10
        details.append("Microalbuminuria — early diabetic nephropathy")

    hemo = float(patient.get('hemo') or patient.get('hemoglobin') or 12)
    if hemo < 10:
        score += 10
        details.append("Anemia — suggests chronic disease from diabetes")

    score = _clamp(score, 0, 100)
    level = 'High' if score >= 60 else 'Moderate' if score >= 30 else 'Low'

    return {
        'condition': 'Diabetes Complications',
        'score': score,
        'level': level,
        'details': details,
        'recommendations': _diabetes_recs(score, bg, uacr),
    }


def hypertension_risk_score(patient: dict) -> dict:
    """
    Hypertension severity and organ damage risk score (0-100).
    """
    score = 0
    details = []

    bp = float(patient.get('bp') or patient.get('blood_pressure_systolic') or 130)
    if bp >= 180:
        score += 40
        details.append(f"Hypertensive crisis (BP {bp:.0f} mmHg)")
    elif bp >= 160:
        score += 25
        details.append(f"Stage 2 hypertension (BP {bp:.0f} mmHg)")
    elif bp >= 140:
        score += 15
        details.append(f"Stage 1 hypertension (BP {bp:.0f} mmHg)")
    elif bp >= 130:
        score += 5

    uacr = float(patient.get('uacr') or 0)
    if uacr > 300:
        score += 20
        details.append("Macro-albuminuria — hypertensive nephropathy")
    elif uacr > 30:
        score += 10
        details.append("Micro-albuminuria — early hypertensive kidney damage")

    age = float(patient.get('age', 55))
    if age > 65:
        score += 10

    cad = str(patient.get('cad', 'no')).lower() in ('yes', '1', 'true')
    dm = str(patient.get('dm', 'no')).lower() in ('yes', '1', 'true')
    if cad:
        score += 15
        details.append("Existing CAD — hypertension management critical")
    if dm:
        score += 10
        details.append("Diabetes + hypertension — high CKD progression risk")

    score = _clamp(score, 0, 100)
    level = 'High' if score >= 60 else 'Moderate' if score >= 30 else 'Low'

    return {
        'condition': 'Hypertension',
        'score': score,
        'level': level,
        'details': details,
        'recommendations': _htn_recs(score, bp),
    }


def cardiovascular_risk_score(patient: dict) -> dict:
    """
    Simplified Framingham-style cardiovascular risk (10-year ASCVD risk %).
    Returns both percentage risk and categorical risk.
    """
    age = float(patient.get('age', 55))
    bp = float(patient.get('bp') or patient.get('blood_pressure_systolic') or 130)
    hemo = float(patient.get('hemo') or patient.get('hemoglobin') or 12)
    dm = str(patient.get('dm', 'no')).lower() in ('yes', '1', 'true')
    htn = str(patient.get('htn', 'no')).lower() in ('yes', '1', 'true')
    cad = str(patient.get('cad', 'no')).lower() in ('yes', '1', 'true')
    egfr = float(patient.get('egfr') or patient.get('sc') or 45)

    # Simplified log-hazard
    log_risk = (
        0.064 * age
        + 0.019 * bp
        - 0.150 * hemo
        + 0.661 * int(dm)
        + 0.491 * int(htn)
        + 0.893 * int(cad)
        - 0.008 * egfr
        - 6.0
    )
    raw_risk = 1 / (1 + math.exp(-log_risk))
    ten_year_risk_pct = round(_clamp(raw_risk * 100, 1, 99), 1)

    if ten_year_risk_pct >= 20:
        category = 'Very High'
    elif ten_year_risk_pct >= 10:
        category = 'High'
    elif ten_year_risk_pct >= 5:
        category = 'Moderate'
    else:
        category = 'Low'

    details = []
    if cad:
        details.append("Existing CAD — secondary prevention required")
    if dm and htn:
        details.append("Diabetes + Hypertension — major CV risk multiplier")
    if egfr < 30:
        details.append("Severe CKD independently doubles cardiovascular risk")
    elif egfr < 60:
        details.append("CKD adds significant cardiovascular risk")

    return {
        'condition': 'Cardiovascular Disease (10-year risk)',
        'score': ten_year_risk_pct,
        'level': category,
        'details': details,
        'recommendations': _cv_recs(ten_year_risk_pct, cad),
    }


def composite_comorbidity_score(patient: dict) -> dict:
    """Returns all three comorbidity scores plus an overall composite."""
    dm_score = diabetes_risk_score(patient)
    htn_score = hypertension_risk_score(patient)
    cv_score = cardiovascular_risk_score(patient)

    weights = {'diabetes': 0.30, 'hypertension': 0.30, 'cardiovascular': 0.40}
    composite = round(
        dm_score['score'] * weights['diabetes']
        + htn_score['score'] * weights['hypertension']
        + cv_score['score'] * weights['cardiovascular'],
        1,
    )
    overall_level = 'Critical' if composite >= 70 else 'High' if composite >= 50 else 'Moderate' if composite >= 30 else 'Low'

    return {
        'composite_score': composite,
        'overall_level': overall_level,
        'scores': {
            'diabetes': dm_score,
            'hypertension': htn_score,
            'cardiovascular': cv_score,
        },
    }


def _diabetes_recs(score, bg, uacr):
    recs = []
    if bg > 200:
        recs.append("Urgent glycemic optimization — consider insulin or GLP-1 agonist")
    if uacr > 300:
        recs.append("Start ACE inhibitor / ARB for diabetic nephropathy protection")
    if score >= 60:
        recs.append("Endocrinology referral; consider SGLT2 inhibitor (if eGFR ≥20)")
    recs.append("HbA1c target < 7% (< 8% if elderly or at hypoglycemia risk)")
    return recs


def _htn_recs(score, bp):
    recs = []
    if bp >= 180:
        recs.append("EMERGENCY: Hypertensive crisis — immediate antihypertensive therapy")
    elif bp >= 160:
        recs.append("Add second antihypertensive agent; ensure ACE-I or ARB is included")
    elif bp >= 140:
        recs.append("Intensify antihypertensive regimen; BP target < 130/80 mmHg in CKD")
    recs.append("Dietary sodium restriction < 2g/day")
    recs.append("BP monitoring at every visit; home BP monitoring encouraged")
    return recs


def _cv_recs(risk_pct, cad):
    recs = []
    if cad or risk_pct >= 20:
        recs.append("High-intensity statin therapy (atorvastatin 40-80mg or rosuvastatin 20-40mg)")
        recs.append("Aspirin 81mg daily (if not contraindicated)")
        recs.append("Cardiology referral for secondary prevention optimization")
    elif risk_pct >= 10:
        recs.append("Moderate-intensity statin therapy")
        recs.append("Consider cardiac stress testing if symptomatic")
    recs.append("Aerobic exercise 150 min/week (adapted to functional status)")
    recs.append("Smoking cessation if applicable")
    return recs
