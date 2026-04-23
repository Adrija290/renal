"""
Clinical treatment recommendation engine.
Stage-based guideline-driven suggestions (KDIGO 2022 aligned).
"""
from models.gfr_forecaster import classify_ckd_stage


def generate_treatment_plan(patient: dict, egfr: float, uacr: float = None) -> dict:
    """
    Generate a stage-specific treatment plan.
    patient: dict with clinical features.
    egfr: current eGFR in mL/min/1.73m².
    uacr: urine albumin-to-creatinine ratio in mg/g.
    """
    stage = classify_ckd_stage(egfr)
    dm = str(patient.get('dm', 'no')).lower() in ('yes', '1', 'true')
    htn = str(patient.get('htn', 'no')).lower() in ('yes', '1', 'true')
    cad = str(patient.get('cad', 'no')).lower() in ('yes', '1', 'true')
    anemia = str(patient.get('ane', 'no')).lower() in ('yes', '1', 'true')
    hemo = float(patient.get('hemo') or 12)
    potassium = float(patient.get('pot') or patient.get('potassium') or 4.5)
    uacr = uacr or float(patient.get('al') or 0) * 300  # rough proxy from albumin

    recommendations = []
    urgent_actions = []
    medications = []
    monitoring = []
    referrals = []

    # ── BP management ──────────────────────────────────────────────
    if htn or egfr < 60:
        target_bp = '< 120/80 mmHg' if uacr > 30 else '< 130/80 mmHg'
        medications.append({
            'class': 'ACE Inhibitor / ARB',
            'example': 'Ramipril or Losartan',
            'dose_note': 'First-line for CKD with proteinuria; monitor K+ and creatinine at 2 weeks',
            'target': target_bp,
        })
        recommendations.append(f"BP target: {target_bp} (KDIGO 2022). "
                                f"ACE-I or ARB preferred for renoprotection.")

    # ── Diabetes management ────────────────────────────────────────
    if dm:
        if egfr >= 30:
            medications.append({
                'class': 'SGLT2 Inhibitor',
                'example': 'Empagliflozin 10mg or Dapagliflozin 10mg',
                'dose_note': 'Cardio-renal protection; do not initiate below eGFR 20',
                'target': 'eGFR preservation, HbA1c 7-8%',
            })
        if uacr > 30 or egfr < 60:
            medications.append({
                'class': 'Finerenone (MRA)',
                'example': 'Finerenone 10-20mg',
                'dose_note': 'Non-steroidal MRA; reduces CKD progression in DKD; K+ >5 is contraindication',
                'target': 'Reduce proteinuria and CKD progression',
            })
        if egfr >= 30:
            medications.append({
                'class': 'GLP-1 Receptor Agonist',
                'example': 'Semaglutide or Liraglutide',
                'dose_note': 'Cardiovascular benefit; weight-independent renal benefit',
                'target': 'HbA1c + CV risk reduction',
            })
        recommendations.append("HbA1c target 7-8%; hypoglycemia risk increases with CKD progression.")
        recommendations.append("Avoid metformin if eGFR < 30 (lactic acidosis risk).")

    # ── Anemia management ─────────────────────────────────────────
    if anemia or hemo < 10:
        urgent_actions.append(f"Anemia detected (Hgb {hemo:.1f} g/dL). Check iron studies, B12, folate.")
        if hemo < 9:
            urgent_actions.append("Hgb critically low — consider ESA therapy and IV iron supplementation.")
            medications.append({
                'class': 'ESA (Erythropoiesis-Stimulating Agent)',
                'example': 'Darbepoetin alfa or Epoetin alfa',
                'dose_note': 'Target Hgb 10-11.5 g/dL; check iron saturation before initiating',
                'target': 'Hgb 10-11.5 g/dL',
            })
        medications.append({
            'class': 'IV Iron',
            'example': 'Ferric carboxymaltose or Iron sucrose',
            'dose_note': 'Correct iron deficiency before ESA; TSAT target ≥20%, ferritin 200-500',
            'target': 'Iron repletion',
        })

    # ── Hyperkalemia management ───────────────────────────────────
    if potassium > 5.0:
        if potassium >= 6.0:
            urgent_actions.append(f"CRITICAL: Hyperkalemia K+ {potassium} mmol/L — immediate medical management required.")
        else:
            urgent_actions.append(f"Elevated potassium {potassium} mmol/L — dietary restriction, review ACE-I dose.")
        medications.append({
            'class': 'Potassium Binder',
            'example': 'Patiromer or Sodium Zirconium Cyclosilicate',
            'dose_note': 'Use if K+ persistently >5.0 on optimal RAASi therapy',
            'target': 'K+ < 5.0 mmol/L',
        })

    # ── Mineral bone disease ──────────────────────────────────────
    if egfr < 45:
        recommendations.append("Monitor PTH, phosphorus, calcium every 3-6 months (CKD-MBD).")
        medications.append({
            'class': 'Active Vitamin D',
            'example': 'Calcitriol 0.25-0.5 mcg daily',
            'dose_note': 'For secondary hyperparathyroidism when 25-OH-D corrected',
            'target': 'PTH within 2-9x upper normal',
        })
        if egfr < 30:
            medications.append({
                'class': 'Phosphate Binder',
                'example': 'Sevelamer carbonate or Calcium carbonate',
                'dose_note': 'With meals; avoid calcium-based binders if hypercalcemia',
                'target': 'Phosphorus 3.5-5.5 mg/dL',
            })

    # ── Statin / CV prevention ────────────────────────────────────
    if egfr < 60 or cad:
        medications.append({
            'class': 'Statin',
            'example': 'Atorvastatin 40-80mg',
            'dose_note': 'Statin or statin+ezetimibe combination preferred; SHARP trial supports use',
            'target': 'LDL < 70 mg/dL (< 55 if very high CV risk)',
        })

    # ── Stage-specific referrals ──────────────────────────────────
    if stage in ('G4', 'G5'):
        referrals.append("Nephrology: Immediate referral for ESRD preparation and RRT planning")
        referrals.append("Vascular Surgery: AV fistula planning (minimum 6 months before dialysis)")
        referrals.append("Transplant center: Evaluate for pre-emptive kidney transplant listing")
        referrals.append("Palliative care: Advance care planning discussion")
    elif stage == 'G3b':
        referrals.append("Nephrology: Referral recommended for optimizing CKD management")
        referrals.append("Dietitian: Stage-specific dietary counseling")
    elif stage == 'G3a':
        referrals.append("Consider nephrology referral if rapid progression, young age, or uncertain diagnosis")

    # ── Monitoring schedule ───────────────────────────────────────
    freq_map = {'G1': '12mo', 'G2': '12mo', 'G3a': '6mo', 'G3b': '6mo', 'G4': '3mo', 'G5': '1-3mo'}
    monitoring = [
        f"eGFR and urine ACR: every {freq_map.get(stage, '6mo')}",
        "Blood pressure at every visit",
        "Electrolytes (K+, Na+, bicarbonate) aligned with eGFR monitoring",
        "CBC for anemia monitoring",
        "Lipid panel annually",
    ]
    if egfr < 45:
        monitoring.append("PTH, phosphorus, calcium, 25-OH-D: every 3-6 months")
    if egfr < 30:
        monitoring.append("Acid-base status, bicarbonate: every 3 months")

    return {
        'stage': stage,
        'egfr': egfr,
        'urgent_actions': urgent_actions,
        'medications': medications,
        'recommendations': recommendations,
        'monitoring': monitoring,
        'referrals': referrals,
        'guideline': 'KDIGO CKD Clinical Practice Guideline 2022',
    }
