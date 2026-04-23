"""
Personal baseline anomaly detection.
Each patient has their own rolling mean and std for every lab value.
Alerts when a new value deviates beyond Z-score threshold.
"""
import numpy as np
from config import Config

LAB_REFERENCE = {
    'egfr': {'label': 'eGFR', 'unit': 'mL/min/1.73m²', 'low': 60, 'high': 120},
    'creatinine': {'label': 'Serum Creatinine', 'unit': 'mg/dL', 'low': 0.6, 'high': 1.2},
    'albumin': {'label': 'Albumin', 'unit': 'g/dL', 'low': 3.5, 'high': 5.0},
    'hemoglobin': {'label': 'Hemoglobin', 'unit': 'g/dL', 'low': 12.0, 'high': 17.5},
    'potassium': {'label': 'Potassium', 'unit': 'mmol/L', 'low': 3.5, 'high': 5.0},
    'sodium': {'label': 'Sodium', 'unit': 'mmol/L', 'low': 135, 'high': 145},
    'blood_pressure_systolic': {'label': 'Systolic BP', 'unit': 'mmHg', 'low': 90, 'high': 130},
    'blood_glucose': {'label': 'Blood Glucose', 'unit': 'mg/dL', 'low': 70, 'high': 100},
    'uacr': {'label': 'Urine ACR', 'unit': 'mg/g', 'low': 0, 'high': 30},
}

CRITICAL_FLAGS = {
    'potassium': {'critical_high': 6.0, 'critical_low': 3.0},
    'sodium': {'critical_high': 150, 'critical_low': 125},
    'blood_pressure_systolic': {'critical_high': 180, 'critical_low': 80},
    'hemoglobin': {'critical_high': None, 'critical_low': 7.0},
    'blood_glucose': {'critical_high': 400, 'critical_low': 60},
}


def detect_anomalies(lab_history: list, window: int = 6) -> list:
    """
    lab_history: list of LabResult model objects ordered chronologically.
    window: number of previous readings to establish baseline.
    Returns list of anomaly dicts for the most recent reading.
    """
    if len(lab_history) < 2:
        return []

    latest = lab_history[-1]
    baseline_results = lab_history[-(window + 1):-1]

    anomalies = []

    for field, meta in LAB_REFERENCE.items():
        latest_val = getattr(latest, field, None)
        if latest_val is None:
            continue

        baseline_vals = [getattr(r, field) for r in baseline_results
                         if getattr(r, field) is not None]

        if not baseline_vals:
            anomaly = _check_absolute_range(field, latest_val, meta)
            if anomaly:
                anomalies.append(anomaly)
            continue

        mean = np.mean(baseline_vals)
        std = np.std(baseline_vals)

        if std < 0.001:
            std = mean * 0.05 if mean != 0 else 0.1

        z_score = (latest_val - mean) / std

        if abs(z_score) >= Config.ANOMALY_ZSCORE_THRESHOLD:
            direction = 'elevated' if z_score > 0 else 'reduced'
            severity = _compute_severity(field, latest_val, z_score)
            anomalies.append({
                'field': field,
                'label': meta['label'],
                'unit': meta['unit'],
                'current_value': round(latest_val, 2),
                'personal_mean': round(mean, 2),
                'z_score': round(z_score, 2),
                'direction': direction,
                'severity': severity,
                'message': (
                    f"{meta['label']} is {direction} compared to your personal baseline "
                    f"({latest_val:.1f} vs mean {mean:.1f} {meta['unit']}; "
                    f"Z={z_score:.1f})"
                ),
            })

        # Critical absolute value check (regardless of personal baseline)
        crit = CRITICAL_FLAGS.get(field)
        if crit:
            if crit.get('critical_high') and latest_val >= crit['critical_high']:
                anomalies.append({
                    'field': field,
                    'label': meta['label'],
                    'unit': meta['unit'],
                    'current_value': round(latest_val, 2),
                    'severity': 'critical',
                    'direction': 'critically_elevated',
                    'message': (f"CRITICAL: {meta['label']} critically elevated at "
                                f"{latest_val:.1f} {meta['unit']}. Immediate action required."),
                })
            elif crit.get('critical_low') and latest_val <= crit['critical_low']:
                anomalies.append({
                    'field': field,
                    'label': meta['label'],
                    'unit': meta['unit'],
                    'current_value': round(latest_val, 2),
                    'severity': 'critical',
                    'direction': 'critically_reduced',
                    'message': (f"CRITICAL: {meta['label']} critically low at "
                                f"{latest_val:.1f} {meta['unit']}. Immediate action required."),
                })

    return anomalies


def _check_absolute_range(field: str, value: float, meta: dict) -> dict | None:
    low, high = meta.get('low'), meta.get('high')
    if low is not None and value < low:
        return {
            'field': field, 'label': meta['label'], 'unit': meta['unit'],
            'current_value': value, 'severity': 'moderate', 'direction': 'below_normal',
            'message': f"{meta['label']} is below normal range ({value:.1f} < {low} {meta['unit']})",
        }
    if high is not None and value > high:
        return {
            'field': field, 'label': meta['label'], 'unit': meta['unit'],
            'current_value': value, 'severity': 'moderate', 'direction': 'above_normal',
            'message': f"{meta['label']} is above normal range ({value:.1f} > {high} {meta['unit']})",
        }
    return None


def _compute_severity(field: str, value: float, z_score: float) -> str:
    if abs(z_score) >= 4.0:
        return 'critical'
    if abs(z_score) >= 3.0:
        return 'high'
    return 'moderate'
