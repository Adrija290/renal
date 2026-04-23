"""
Patient-facing personal risk score dashboard.
Translates complex clinical data into plain-language risk summaries.
"""
from models.gfr_forecaster import classify_ckd_stage

RISK_BANDS = [
    (80, 100, 'Very High', '#dc3545', 'Your kidney health requires immediate medical attention.'),
    (60, 80, 'High', '#fd7e14', 'Your kidneys are significantly affected. Close monitoring needed.'),
    (40, 60, 'Moderate', '#ffc107', 'Your kidney health needs attention and lifestyle changes.'),
    (20, 40, 'Low-Moderate', '#20c997', 'Your kidneys are mildly affected. Preventive care is working.'),
    (0, 20, 'Low', '#28a745', 'Your kidney health is generally good. Keep up healthy habits.'),
]

PLAIN_LANGUAGE_STAGE = {
    'G1': "Your kidneys are working normally. The goal is to keep them that way.",
    'G2': "Your kidneys are slightly reduced in function. Early-stage monitoring is key.",
    'G3a': "Your kidneys are working at about half their normal capacity. Treatment can slow this.",
    'G3b': "Your kidney function is moderately to severely reduced. Active management is needed.",
    'G4': "Your kidneys are severely reduced. Preparing for dialysis or transplant is important.",
    'G5': "Kidney failure — dialysis or transplant is needed to sustain life.",
}


def build_patient_dashboard(patient: dict, lab_results: list,
                              ckd_probability: float, egfr: float) -> dict:
    """
    Build a patient-facing dashboard summary.
    patient: Patient model object (or dict).
    lab_results: list of LabResult objects.
    ckd_probability: 0-1 from ensemble model.
    egfr: most recent eGFR value.
    """
    stage = classify_ckd_stage(egfr)
    risk_score = round(ckd_probability * 100, 1)

    # Find risk band
    band_label, band_color, band_message = 'Unknown', '#6c757d', ''
    for lo, hi, label, color, msg in RISK_BANDS:
        if lo <= risk_score < hi:
            band_label, band_color, band_message = label, color, msg
            break

    # Trend: compare last 2 lab results
    trend = 'stable'
    trend_message = "Your kidney function has been stable recently."
    if len(lab_results) >= 2:
        egfr_prev = lab_results[-2].egfr
        egfr_curr = lab_results[-1].egfr
        if egfr_curr and egfr_prev:
            delta = egfr_curr - egfr_prev
            if delta < -5:
                trend = 'worsening_fast'
                trend_message = f"Your kidney function has declined significantly ({delta:+.1f}) since your last visit. Please contact your doctor."
            elif delta < -2:
                trend = 'worsening'
                trend_message = f"Your kidney function has declined slightly ({delta:+.1f}). Discuss with your care team."
            elif delta > 2:
                trend = 'improving'
                trend_message = f"Your kidney function improved slightly ({delta:+.1f}). Keep up the good work!"

    # Key metrics for display
    latest = lab_results[-1] if lab_results else None
    key_metrics = []
    if latest:
        key_metrics = [
            {'label': 'Kidney Function (eGFR)', 'value': f"{egfr:.0f}", 'unit': 'mL/min/1.73m²',
             'status': _metric_status('egfr', egfr)},
            {'label': 'Protein in Urine (ACR)', 'value': f"{latest.uacr:.0f}" if latest.uacr else 'N/A',
             'unit': 'mg/g', 'status': _metric_status('uacr', latest.uacr)},
            {'label': 'Blood Pressure', 'value': f"{latest.blood_pressure_systolic}/{latest.blood_pressure_diastolic}",
             'unit': 'mmHg', 'status': _metric_status('bp', latest.blood_pressure_systolic)},
            {'label': 'Hemoglobin', 'value': f"{latest.hemoglobin:.1f}" if latest.hemoglobin else 'N/A',
             'unit': 'g/dL', 'status': _metric_status('hemo', latest.hemoglobin)},
            {'label': 'Potassium', 'value': f"{latest.potassium:.1f}" if latest.potassium else 'N/A',
             'unit': 'mmol/L', 'status': _metric_status('potassium', latest.potassium)},
        ]

    # Goals for patient
    goals = _generate_goals(stage, latest, patient)

    # Next steps in plain language
    next_steps = _plain_language_next_steps(stage, risk_score, trend)

    return {
        'risk_score': risk_score,
        'risk_band': band_label,
        'risk_color': band_color,
        'risk_message': band_message,
        'stage': stage,
        'stage_explanation': PLAIN_LANGUAGE_STAGE.get(stage, ''),
        'trend': trend,
        'trend_message': trend_message,
        'key_metrics': key_metrics,
        'goals': goals,
        'next_steps': next_steps,
        'encouragement': _get_encouragement(trend, risk_score),
    }


def _metric_status(metric: str, value) -> str:
    if value is None:
        return 'unknown'
    thresholds = {
        'egfr': [(60, 'normal'), (30, 'caution'), (0, 'danger')],
        'uacr': [(30, 'normal'), (300, 'caution'), (float('inf'), 'danger')],
        'bp': [(130, 'normal'), (160, 'caution'), (float('inf'), 'danger')],
        'hemo': [(12, 'danger'), (10, 'danger'), (float('inf'), 'normal')],
        'potassium': [(5.0, 'normal'), (5.5, 'caution'), (float('inf'), 'danger')],
    }
    checks = thresholds.get(metric, [])
    for threshold, status in checks:
        if metric in ('hemo',):
            if value < threshold:
                return status
        else:
            if value < threshold:
                return status
    return 'normal'


def _generate_goals(stage: str, latest, patient) -> list:
    goals = [
        {'goal': 'Keep BP below 130/80 mmHg', 'importance': 'critical',
         'tip': 'Take medications as prescribed; limit salt; check BP daily if possible.'},
        {'goal': 'Attend all scheduled lab checks', 'importance': 'high',
         'tip': 'Catching changes early makes treatment more effective.'},
        {'goal': 'Follow your kidney-friendly diet', 'importance': 'high',
         'tip': 'Ask your dietitian about low-potassium, low-phosphorus food choices.'},
    ]
    if stage in ('G3a', 'G3b', 'G4', 'G5'):
        goals.append({'goal': 'Protect your arm veins', 'importance': 'moderate',
                      'tip': 'Avoid blood draws and IV lines in your non-dominant arm — these veins may be needed for dialysis access.'})
    if latest and latest.hemoglobin and latest.hemoglobin < 11:
        goals.append({'goal': 'Treat your anemia', 'importance': 'high',
                      'tip': 'Low blood count makes you tired. Ask your doctor about iron or EPO therapy.'})
    return goals


def _plain_language_next_steps(stage: str, risk_score: float, trend: str) -> list:
    steps = []
    if trend in ('worsening', 'worsening_fast') or risk_score > 60:
        steps.append("Schedule an urgent appointment with your kidney specialist (nephrologist).")
    if stage in ('G4', 'G5'):
        steps.append("Ask your doctor about dialysis and kidney transplant options — planning ahead matters.")
        steps.append("Get a vein mapping test to prepare for dialysis access (AV fistula).")
    steps.append("Review your medications — some common drugs can hurt your kidneys.")
    steps.append("Log your symptoms below — we track how fatigue and swelling connect to your labs.")
    return steps


def _get_encouragement(trend: str, risk_score: float) -> str:
    if trend == 'improving':
        return "Your recent labs show improvement — your efforts are making a difference!"
    if trend == 'stable' and risk_score < 50:
        return "Your kidney function has been stable. Keep following your care plan."
    if risk_score > 70:
        return "Managing kidney disease is challenging, but you're not alone. Your care team is here to help."
    return "Small daily choices — diet, medications, monitoring — protect your kidneys long-term."
