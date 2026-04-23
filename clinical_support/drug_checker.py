"""
Nephrotoxic drug checker with GFR-based dose adjustment.
Cross-checks a patient's medication list against the nephrotoxic drugs database.
"""
import json
import os
from config import Config


def _load_drug_db() -> dict:
    path = os.path.join(Config.REFERENCE_DIR, 'nephrotoxic_drugs.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def check_medications(medication_list: list[str], egfr: float) -> dict:
    """
    Check a list of medication names against nephrotoxic drug database.
    Returns alerts, dose adjustments, and contraindications.
    """
    db = _load_drug_db()
    alerts = []
    dose_adjustments = []
    contraindicated = []
    safe_alternatives = {}

    medication_lower = [m.strip().lower() for m in medication_list]

    for category, cat_data in db.get('categories', {}).items():
        for drug in cat_data.get('drugs', []):
            drug_name = drug['name'].lower()
            brand_names = [b.lower() for b in drug.get('brand', [])]
            all_names = [drug_name] + brand_names

            matched = any(
                any(med_name in drug_alias or drug_alias in med_name
                    for drug_alias in all_names)
                for med_name in medication_lower
            )

            if not matched:
                continue

            avoid_gfr = drug.get('avoid_below_gfr')
            dose_note = drug.get('dose_adjustment', '')

            if avoid_gfr and egfr < avoid_gfr:
                contraindicated.append({
                    'drug': drug['name'],
                    'category': category.replace('_', ' '),
                    'risk_level': cat_data['risk_level'],
                    'reason': f"Contraindicated when eGFR < {avoid_gfr} (current: {egfr:.0f})",
                    'mechanism': cat_data['mechanism'],
                    'action': dose_note,
                })
                alt = db.get('safe_alternatives', {}).get(category)
                if alt:
                    safe_alternatives[drug['name']] = alt
            else:
                severity = _risk_to_severity(cat_data['risk_level'], egfr, avoid_gfr)
                alert = {
                    'drug': drug['name'],
                    'category': category.replace('_', ' '),
                    'risk_level': cat_data['risk_level'],
                    'mechanism': cat_data['mechanism'],
                    'dose_adjustment': dose_note,
                }
                if severity == 'warning':
                    alerts.append(alert)
                else:
                    dose_adjustments.append(alert)

    # Auto-adjust dosing for renally-cleared drugs
    auto_adjustments = _auto_dose_adjustments(medication_lower, egfr)

    return {
        'egfr_used': egfr,
        'drugs_checked': len(medication_list),
        'contraindicated': contraindicated,
        'alerts': alerts,
        'dose_adjustments': dose_adjustments + auto_adjustments,
        'safe_alternatives': safe_alternatives,
        'total_concerns': len(contraindicated) + len(alerts),
    }


def _risk_to_severity(risk_level: str, egfr: float, avoid_gfr) -> str:
    if risk_level == 'high':
        return 'warning'
    if risk_level == 'moderate' and egfr < 45:
        return 'warning'
    return 'info'


def _auto_dose_adjustments(medications: list[str], egfr: float) -> list[dict]:
    """
    Auto-compute dose adjustments for common renally-cleared drugs.
    Uses Cockcroft-Gault derived guidance.
    """
    adjustments = []
    rules = [
        ('metformin',   [(45, 'Reduce dose; use 50% of standard; monitor closely'),
                         (30, 'CONTRAINDICATED — stop immediately')]),
        ('lisinopril',  [(30, 'Reduce starting dose; titrate cautiously; monitor K+')]),
        ('enalapril',   [(30, 'Start at 2.5mg; titrate slowly; monitor K+')]),
        ('gabapentin',  [(60, 'Reduce to 600-1200mg/day'),
                         (30, 'Reduce to 300mg/day, extended intervals'),
                         (15, 'Max 300mg every other day')]),
        ('digoxin',     [(60, 'Reduce dose 25%; monitor levels'),
                         (30, 'Reduce dose 50%; monitor closely')]),
        ('atenolol',    [(35, 'Reduce dose 50% or extend interval')]),
        ('ranitidine',  [(50, 'Reduce dose 50%')]),
        ('allopurinol', [(50, 'Reduce to max 200mg/day'),
                         (30, 'Reduce to max 100mg/day')]),
    ]
    for med, thresholds in rules:
        if any(med in m for m in medications):
            for gfr_thresh, note in thresholds:
                if egfr < gfr_thresh:
                    adjustments.append({
                        'drug': med.capitalize(),
                        'category': 'Auto-adjustment',
                        'risk_level': 'moderate',
                        'mechanism': 'Renal clearance reduced',
                        'dose_adjustment': note,
                    })
                    break
    return adjustments


def flag_drug_interactions_with_ckd(medications: list[str], egfr: float,
                                     potassium: float = None) -> list[dict]:
    """Additional drug-disease interaction checks specific to CKD."""
    flags = []
    meds_lower = [m.lower() for m in medications]

    # Dual RAASi blockade is high-risk in CKD
    acei_list = ['lisinopril', 'enalapril', 'ramipril', 'benazepril', 'captopril', 'perindopril']
    arb_list = ['losartan', 'valsartan', 'irbesartan', 'candesartan', 'telmisartan']
    has_acei = any(any(a in m for a in acei_list) for m in meds_lower)
    has_arb = any(any(a in m for a in arb_list) for m in meds_lower)
    if has_acei and has_arb:
        flags.append({
            'type': 'drug_interaction',
            'severity': 'critical',
            'message': 'ACE inhibitor + ARB combination: AVOID in CKD (ONTARGET trial) — increased AKI and hyperkalemia risk.',
        })

    # NSAIDs + ACE-I + Diuretics = Triple Whammy
    has_nsaid = any(n in m for m in meds_lower for n in ['ibuprofen', 'naproxen', 'diclofenac', 'indomethacin'])
    has_diuretic = any(d in m for m in meds_lower for d in ['furosemide', 'hydrochlorothiazide', 'bumetanide'])
    if has_nsaid and (has_acei or has_arb) and has_diuretic:
        flags.append({
            'type': 'triple_whammy',
            'severity': 'critical',
            'message': 'TRIPLE WHAMMY: NSAIDs + RAASi + Diuretic significantly increases AKI risk. Stop NSAID immediately.',
        })

    # Potassium-sparing + high K+
    if potassium and potassium > 5.0:
        k_sparing = ['spironolactone', 'eplerenone', 'amiloride', 'triamterene']
        if any(any(k in m for k in k_sparing) for m in meds_lower):
            flags.append({
                'type': 'hyperkalemia_risk',
                'severity': 'high',
                'message': f'K+ {potassium} mmol/L with potassium-sparing agent — high hyperkalemia risk. Consider stopping K+-sparing drug.',
            })

    return flags
