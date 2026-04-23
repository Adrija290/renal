"""
Population cohort analytics: risk stratification, geographic patterns,
demographic breakdowns, and outcome statistics.
"""
import numpy as np
import pandas as pd
from models.gfr_forecaster import classify_ckd_stage


def stratify_by_stage(patients: list, lab_results_map: dict) -> dict:
    """
    Stratify patient population by CKD stage.
    patients: list of Patient model objects.
    lab_results_map: dict {patient_id: [LabResult, ...]}
    """
    stage_counts = {'G1': 0, 'G2': 0, 'G3a': 0, 'G3b': 0, 'G4': 0, 'G5': 0, 'Unknown': 0}
    stage_demographics = {s: {'ages': [], 'diabetes': 0, 'hypertension': 0} for s in stage_counts}

    for patient in patients:
        labs = lab_results_map.get(patient.id, [])
        latest_lab = max(labs, key=lambda l: l.date, default=None) if labs else None
        if latest_lab and latest_lab.egfr:
            stage = classify_ckd_stage(float(latest_lab.egfr))
        else:
            stage = 'Unknown'

        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if stage != 'Unknown':
            demo = stage_demographics[stage]
            if patient.age:
                demo['ages'].append(patient.age)
            if patient.diabetes:
                demo['diabetes'] += 1
            if patient.hypertension:
                demo['hypertension'] += 1

    # Compute averages
    stage_summary = {}
    total_known = sum(v for k, v in stage_counts.items() if k != 'Unknown')
    for stage, count in stage_counts.items():
        if stage == 'Unknown':
            continue
        demo = stage_demographics[stage]
        stage_summary[stage] = {
            'count': count,
            'percentage': round(count / max(total_known, 1) * 100, 1),
            'avg_age': round(np.mean(demo['ages']), 1) if demo['ages'] else None,
            'diabetes_rate': round(demo['diabetes'] / max(count, 1) * 100, 1),
            'hypertension_rate': round(demo['hypertension'] / max(count, 1) * 100, 1),
        }

    return {
        'stage_distribution': stage_summary,
        'total_patients': len(patients),
        'stage_counts': {k: v for k, v in stage_counts.items() if k != 'Unknown'},
        'unknown_stage': stage_counts['Unknown'],
    }


def demographic_risk_analysis(patients: list, lab_results_map: dict) -> dict:
    """Analyze CKD risk across age groups, sex, and comorbidities."""
    groups = {
        'age': {'<40': [], '40-59': [], '60-74': [], '75+': []},
        'sex': {'M': [], 'F': []},
        'diabetes': {'Yes': [], 'No': []},
        'hypertension': {'Yes': [], 'No': []},
    }

    for patient in patients:
        labs = lab_results_map.get(patient.id, [])
        latest = max(labs, key=lambda l: l.date, default=None) if labs else None
        egfr = float(latest.egfr) if latest and latest.egfr else None

        if egfr is None:
            continue

        age = patient.age or 0
        if age < 40:
            groups['age']['<40'].append(egfr)
        elif age < 60:
            groups['age']['40-59'].append(egfr)
        elif age < 75:
            groups['age']['60-74'].append(egfr)
        else:
            groups['age']['75+'].append(egfr)

        sex_key = patient.sex if patient.sex in ('M', 'F') else 'F'
        groups['sex'][sex_key].append(egfr)

        dm_key = 'Yes' if patient.diabetes else 'No'
        groups['diabetes'][dm_key].append(egfr)

        htn_key = 'Yes' if patient.hypertension else 'No'
        groups['hypertension'][htn_key].append(egfr)

    analysis = {}
    for group_name, subgroups in groups.items():
        analysis[group_name] = {}
        for subgroup, egfr_vals in subgroups.items():
            if not egfr_vals:
                continue
            ckd_count = sum(1 for e in egfr_vals if e < 60)
            analysis[group_name][subgroup] = {
                'n': len(egfr_vals),
                'mean_egfr': round(np.mean(egfr_vals), 1),
                'median_egfr': round(np.median(egfr_vals), 1),
                'ckd_prevalence_pct': round(ckd_count / len(egfr_vals) * 100, 1),
            }

    return analysis


def egfr_decline_statistics(patients: list, lab_results_map: dict) -> dict:
    """Compute population-level eGFR decline statistics."""
    decline_rates = []
    fast_decliners = 0
    stable = 0
    improving = 0

    for patient in patients:
        labs = sorted(lab_results_map.get(patient.id, []),
                       key=lambda l: l.date)
        egfr_vals = [l.egfr for l in labs if l.egfr is not None]

        if len(egfr_vals) < 3:
            continue

        x = np.arange(len(egfr_vals))
        slope = np.polyfit(x, egfr_vals, 1)[0]
        annual_rate = slope * 4

        decline_rates.append(annual_rate)
        if annual_rate < -5:
            fast_decliners += 1
        elif annual_rate > 0:
            improving += 1
        else:
            stable += 1

    if not decline_rates:
        return {'message': 'Insufficient longitudinal data'}

    return {
        'n_with_trend': len(decline_rates),
        'mean_annual_decline': round(np.mean(decline_rates), 2),
        'median_annual_decline': round(np.median(decline_rates), 2),
        'std_annual_decline': round(np.std(decline_rates), 2),
        'fast_decliners_n': fast_decliners,
        'fast_decliners_pct': round(fast_decliners / len(decline_rates) * 100, 1),
        'stable_n': stable,
        'improving_n': improving,
        'decline_distribution': {
            '>-2': sum(1 for r in decline_rates if r > -2),
            '-2 to -5': sum(1 for r in decline_rates if -5 <= r <= -2),
            '<-5': sum(1 for r in decline_rates if r < -5),
        },
    }


def outcome_statistics(patients: list, predictions_list: list) -> dict:
    """Summary statistics across all model predictions."""
    if not predictions_list:
        return {}

    ckd_probs = [p.ckd_probability for p in predictions_list if p.ckd_probability is not None]
    predictions = [p.prediction for p in predictions_list if p.prediction is not None]

    return {
        'total_predictions': len(predictions_list),
        'ckd_positive_count': sum(1 for p in predictions if p == 1),
        'ckd_negative_count': sum(1 for p in predictions if p == 0),
        'ckd_positive_rate': round(np.mean(predictions) * 100, 1) if predictions else None,
        'mean_ckd_probability': round(np.mean(ckd_probs) * 100, 1) if ckd_probs else None,
        'high_risk_count': sum(1 for p in ckd_probs if p >= 0.7),
        'total_patients': len(patients),
    }
