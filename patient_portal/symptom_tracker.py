"""
Symptom tracker that correlates patient-reported symptoms with lab trends.
Detects patterns like fatigue correlating with anemia or fluid overload with swelling.
"""
import datetime
import numpy as np
from scipy.stats import pearsonr


def correlate_symptoms_with_labs(symptom_entries: list, lab_results: list) -> dict:
    """
    Symptom entries and lab results must have matching timestamps.
    Returns correlation analysis and pattern detection.
    """
    if not symptom_entries or not lab_results:
        return {'correlations': [], 'patterns': [], 'message': 'Insufficient data for correlation analysis'}

    symptoms = sorted(symptom_entries, key=lambda x: x.timestamp)
    labs = sorted(lab_results, key=lambda x: x.date)

    # Build time-aligned arrays
    correlations = []
    corr_pairs = [
        ('fatigue_score', 'hemoglobin', 'Fatigue ↔ Hemoglobin',
         'High fatigue with low hemoglobin strongly suggests anemia as the cause.'),
        ('swelling_score', 'blood_pressure_systolic', 'Swelling ↔ Blood Pressure',
         'Leg swelling with elevated BP suggests fluid overload — review diuretic dose.'),
        ('nausea_score', 'egfr', 'Nausea ↔ eGFR',
         'Nausea worsening as eGFR declines is a uremic symptom — requires urgent review.'),
        ('fatigue_score', 'egfr', 'Fatigue ↔ eGFR',
         'Fatigue tracking with eGFR decline helps quantify uremic burden.'),
        ('swelling_score', 'egfr', 'Swelling ↔ eGFR',
         'Progressive edema with declining eGFR suggests worsening fluid retention.'),
    ]

    for sym_field, lab_field, label, interpretation in corr_pairs:
        sym_vals = [getattr(s, sym_field, None) for s in symptoms]
        sym_dates = [s.timestamp for s in symptoms]

        lab_vals = []
        for sym_date in sym_dates:
            closest_lab = _find_closest_lab(labs, sym_date, lab_field)
            lab_vals.append(closest_lab)

        clean_pairs = [(s, l) for s, l in zip(sym_vals, lab_vals) if s is not None and l is not None]
        if len(clean_pairs) < 3:
            continue

        sv, lv = zip(*clean_pairs)
        try:
            r, p_value = pearsonr(sv, lv)
            strength = 'Strong' if abs(r) >= 0.7 else 'Moderate' if abs(r) >= 0.4 else 'Weak'
            direction = 'positive' if r > 0 else 'negative'
            correlations.append({
                'label': label,
                'correlation_r': round(r, 3),
                'p_value': round(p_value, 4),
                'significant': p_value < 0.05,
                'strength': strength,
                'direction': direction,
                'interpretation': interpretation,
                'data_points': len(clean_pairs),
            })
        except Exception:
            continue

    patterns = _detect_symptom_patterns(symptoms, labs)
    trend_summary = _symptom_trend_summary(symptoms)

    return {
        'correlations': sorted(correlations, key=lambda x: abs(x['correlation_r']), reverse=True),
        'patterns': patterns,
        'trend_summary': trend_summary,
        'total_entries': len(symptoms),
    }


def _find_closest_lab(labs, target_date, field) -> float | None:
    """Find the lab result closest in time to the symptom entry date."""
    if not labs:
        return None
    if isinstance(target_date, datetime.datetime):
        target_ts = target_date.timestamp()
    else:
        target_ts = datetime.datetime.combine(target_date, datetime.time()).timestamp()

    best = None
    best_diff = float('inf')
    for lab in labs:
        lab_date = lab.date
        if isinstance(lab_date, datetime.datetime):
            lab_ts = lab_date.timestamp()
        else:
            lab_ts = datetime.datetime.combine(lab_date, datetime.time()).timestamp()

        diff = abs(lab_ts - target_ts)
        if diff < best_diff:
            val = getattr(lab, field, None)
            if val is not None:
                best_diff = diff
                best = val
    return best


def _detect_symptom_patterns(symptoms: list, labs: list) -> list:
    """Detect clinically meaningful symptom patterns."""
    patterns = []

    if not symptoms:
        return patterns

    recent_symptoms = symptoms[-3:]

    avg_fatigue = np.mean([s.fatigue_score for s in recent_symptoms if s.fatigue_score is not None] or [0])
    avg_swelling = np.mean([s.swelling_score for s in recent_symptoms if s.swelling_score is not None] or [0])
    avg_nausea = np.mean([s.nausea_score for s in recent_symptoms if s.nausea_score is not None] or [0])
    sob_count = sum(1 for s in recent_symptoms if s.shortness_of_breath)
    confusion_count = sum(1 for s in recent_symptoms if s.confusion)

    if avg_fatigue >= 7:
        patterns.append({
            'pattern': 'Severe Fatigue',
            'severity': 'high',
            'message': f"Fatigue score averaging {avg_fatigue:.0f}/10 over recent entries. Check hemoglobin and thyroid function.",
            'action': 'Review blood count; consider iron studies and EPO therapy',
        })
    elif avg_fatigue >= 5:
        patterns.append({
            'pattern': 'Moderate Fatigue',
            'severity': 'moderate',
            'message': f"Consistent moderate fatigue ({avg_fatigue:.0f}/10). Common in CKD — optimizing anemia management helps.",
            'action': 'Discuss with nephrologist at next visit',
        })

    if avg_swelling >= 6:
        patterns.append({
            'pattern': 'Significant Edema',
            'severity': 'high',
            'message': f"Swelling score {avg_swelling:.0f}/10 — possible fluid overload. Check daily weight and BP.",
            'action': 'Review diuretic dose; restrict sodium and fluid; check weight daily',
        })

    if avg_nausea >= 5:
        patterns.append({
            'pattern': 'Uremic Nausea',
            'severity': 'high',
            'message': f"Persistent nausea ({avg_nausea:.0f}/10) in CKD context. May indicate worsening uremia — check BUN.",
            'action': 'Urgent nephrologist review; check BUN/creatinine; assess for dialysis timing',
        })

    if sob_count >= 2:
        patterns.append({
            'pattern': 'Shortness of Breath',
            'severity': 'critical',
            'message': "Repeated episodes of shortness of breath — could indicate fluid overload, pulmonary edema, or anemia.",
            'action': 'Urgent medical evaluation; chest X-ray; check fluid status and hemoglobin',
        })

    if confusion_count >= 1:
        patterns.append({
            'pattern': 'Uremic Encephalopathy Risk',
            'severity': 'critical',
            'message': "Confusion reported — in CKD this may indicate severe uremia. Requires immediate evaluation.",
            'action': 'Emergency evaluation; check BUN, ammonia; assess dialysis initiation',
        })

    # Worsening trend
    if len(symptoms) >= 4:
        early_avg = np.mean([s.fatigue_score for s in symptoms[:2] if s.fatigue_score is not None] or [0])
        late_avg = np.mean([s.fatigue_score for s in symptoms[-2:] if s.fatigue_score is not None] or [0])
        if late_avg - early_avg > 2:
            patterns.append({
                'pattern': 'Worsening Fatigue Trend',
                'severity': 'moderate',
                'message': f"Fatigue worsening over time (from {early_avg:.0f} to {late_avg:.0f}/10). Track alongside lab changes.",
                'action': 'Bring symptom log to next nephrology visit',
            })

    return patterns


def _symptom_trend_summary(symptoms: list) -> dict:
    if not symptoms:
        return {}

    fatigue_vals = [s.fatigue_score for s in symptoms if s.fatigue_score is not None]
    swelling_vals = [s.swelling_score for s in symptoms if s.swelling_score is not None]
    nausea_vals = [s.nausea_score for s in symptoms if s.nausea_score is not None]

    return {
        'avg_fatigue': round(np.mean(fatigue_vals), 1) if fatigue_vals else None,
        'avg_swelling': round(np.mean(swelling_vals), 1) if swelling_vals else None,
        'avg_nausea': round(np.mean(nausea_vals), 1) if nausea_vals else None,
        'total_entries': len(symptoms),
        'sob_episodes': sum(1 for s in symptoms if s.shortness_of_breath),
        'confusion_episodes': sum(1 for s in symptoms if s.confusion),
        'dates': [s.timestamp.strftime('%b %d') for s in symptoms[-8:]],
        'fatigue_series': fatigue_vals[-8:],
        'swelling_series': swelling_vals[-8:],
    }
