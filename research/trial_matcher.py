"""
Clinical trial eligibility matching.
Auto-screens patients against NCT trial eligibility criteria.
"""
import json
import os
from models.gfr_forecaster import classify_ckd_stage
from config import Config


def _load_trials() -> list:
    path = os.path.join(Config.REFERENCE_DIR, 'clinical_trials.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get('trials', [])
    return []


def match_patient_to_trials(patient: dict, egfr: float,
                              uacr: float = None) -> dict:
    """
    Match a patient to eligible clinical trials.
    Returns list of matching trials with eligibility scores.
    """
    trials = _load_trials()
    matches = []
    exclusions = []

    age = float(patient.get('age', 50))
    diabetes = str(patient.get('dm', 'no')).lower() in ('yes', '1', 'true')
    hypertension = str(patient.get('htn', 'no')).lower() in ('yes', '1', 'true')
    cad = str(patient.get('cad', 'no')).lower() in ('yes', '1', 'true')
    albumin = float(patient.get('al', 0))
    uacr = uacr or (albumin * 300)

    for trial in trials:
        elig = trial.get('eligibility', {})
        reasons_met = []
        reasons_failed = []

        # Age check
        min_age = elig.get('min_age')
        max_age = elig.get('max_age')
        if min_age and age < min_age:
            reasons_failed.append(f"Age {age:.0f} < minimum {min_age}")
        elif max_age and age > max_age:
            reasons_failed.append(f"Age {age:.0f} > maximum {max_age}")
        else:
            reasons_met.append(f"Age {age:.0f} within range")

        # eGFR check
        min_egfr = elig.get('min_egfr')
        max_egfr = elig.get('max_egfr')
        if min_egfr and egfr < min_egfr:
            reasons_failed.append(f"eGFR {egfr:.0f} < minimum {min_egfr}")
        elif max_egfr and egfr > max_egfr:
            reasons_failed.append(f"eGFR {egfr:.0f} > maximum {max_egfr}")
        else:
            reasons_met.append(f"eGFR {egfr:.0f} within range")

        # UACR check
        min_uacr = elig.get('min_uacr')
        max_uacr = elig.get('max_uacr')
        if min_uacr and uacr < min_uacr:
            reasons_failed.append(f"UACR {uacr:.0f} < minimum {min_uacr}")
        elif max_uacr and uacr > max_uacr:
            reasons_failed.append(f"UACR {uacr:.0f} > maximum {max_uacr}")
        else:
            if min_uacr or max_uacr:
                reasons_met.append(f"UACR {uacr:.0f} within range")

        # Required conditions
        required_conds = elig.get('required_conditions', [])
        patient_conditions = []
        if diabetes:
            patient_conditions.extend(['Type 2 Diabetes', 'Diabetes'])
        if hypertension:
            patient_conditions.extend(['Hypertension'])
        if cad:
            patient_conditions.extend(['CAD', 'Coronary Artery Disease'])
        patient_conditions.append('CKD')

        for req in required_conds:
            req_lower = req.lower()
            met = any(req_lower in pc.lower() or pc.lower() in req_lower
                       for pc in patient_conditions)
            if met:
                reasons_met.append(f"Has required condition: {req}")
            else:
                reasons_failed.append(f"Missing required condition: {req}")

        # Exclusion conditions
        exclude_conds = elig.get('exclusion_conditions', [])
        for excl in exclude_conds:
            excl_lower = excl.lower()
            if 'dialysis' in excl_lower and egfr < 10:
                reasons_failed.append(f"Exclusion: {excl}")
            elif 'type 1 diabetes' in excl_lower:
                pass  # we don't track T1D separately
            elif 'transplant' in excl_lower:
                pass  # not tracked in this simple model

        score = len(reasons_met) / max(len(reasons_met) + len(reasons_failed), 1)
        eligible = len(reasons_failed) == 0

        trial_result = {
            'nct_id': trial['nct_id'],
            'title': trial['title'],
            'sponsor': trial['sponsor'],
            'phase': trial['phase'],
            'status': trial['status'],
            'conditions': trial['conditions'],
            'eligible': eligible,
            'eligibility_score': round(score * 100, 1),
            'reasons_met': reasons_met,
            'reasons_failed': reasons_failed,
            'primary_endpoint': trial.get('primary_endpoint', ''),
            'contact': trial.get('contact', ''),
        }

        if eligible:
            matches.append(trial_result)
        elif score >= 0.5:
            exclusions.append(trial_result)

    matches.sort(key=lambda x: x['eligibility_score'], reverse=True)
    exclusions.sort(key=lambda x: x['eligibility_score'], reverse=True)

    return {
        'patient_egfr': egfr,
        'patient_uacr': uacr,
        'patient_stage': classify_ckd_stage(egfr),
        'eligible_trials': matches,
        'near_eligible_trials': exclusions[:3],
        'total_screened': len(trials),
        'eligible_count': len(matches),
    }
