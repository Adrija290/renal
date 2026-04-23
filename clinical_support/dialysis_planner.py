"""
Dialysis access preparation timing predictor.
Uses GFR trajectory to recommend optimal AV fistula / PD catheter placement timing.
"""
import datetime
from models.gfr_forecaster import classify_ckd_stage
from config import Config


ACCESS_TYPES = {
    'av_fistula': {
        'label': 'Arteriovenous Fistula (AVF)',
        'preferred_for': 'Hemodialysis — GOLD STANDARD',
        'lead_time_months': 6,
        'maturation_time_months': 3,
        'advantages': ['Lowest infection risk', 'Best long-term patency', 'No catheter required'],
        'disadvantages': ['Requires vein mapping', 'Maturation can fail (~30%)', 'Longer lead time'],
        'eligibility': 'Suitable veins and arteries by vascular mapping',
    },
    'av_graft': {
        'label': 'Arteriovenous Graft (AVG)',
        'preferred_for': 'Hemodialysis — when fistula not feasible',
        'lead_time_months': 4,
        'maturation_time_months': 2,
        'advantages': ['Shorter maturation', 'Larger target for needle insertion'],
        'disadvantages': ['Higher infection risk than AVF', 'Shorter lifespan', 'Thrombosis risk'],
        'eligibility': 'Poor vein quality; previous AVF failure',
    },
    'pd_catheter': {
        'label': 'Peritoneal Dialysis Catheter',
        'preferred_for': 'Peritoneal Dialysis — home-based modality',
        'lead_time_months': 2,
        'maturation_time_months': 1,
        'advantages': ['Home therapy', 'Preserves residual kidney function longer',
                        'No needles required', 'More independence'],
        'disadvantages': ['Peritonitis risk', 'Requires intact peritoneum',
                           'Daily exchanges or overnight cycler'],
        'eligibility': 'Patient motivated for home dialysis; no prior abdominal surgery',
    },
}


def plan_dialysis_access(egfr: float, gfr_trajectory: dict,
                          patient_preferences: dict = None) -> dict:
    """
    Given current eGFR and trajectory forecast, plan optimal dialysis access timing.
    gfr_trajectory: output from GFRForecaster.forecast()
    """
    patient_preferences = patient_preferences or {}

    estimated_months_to_esrd = gfr_trajectory.get('estimated_months_to_esrd')
    annual_decline = gfr_trajectory.get('annual_decline_rate', -3)

    if estimated_months_to_esrd is None:
        if annual_decline and annual_decline < 0 and egfr > 0:
            estimated_months_to_esrd = int((egfr - 10) / abs(annual_decline) * 12)
        else:
            estimated_months_to_esrd = None

    stage = classify_ckd_stage(egfr)
    current_date = datetime.date.today()

    recommendations = []
    action_plan = []

    if stage in ('G4', 'G5') or (estimated_months_to_esrd and estimated_months_to_esrd <= 18):
        urgency = 'urgent'
        priority_message = (
            f"URGENT: eGFR {egfr:.0f} (Stage {stage}). "
            f"Estimated time to ESRD: "
            f"{'~' + str(estimated_months_to_esrd) + ' months' if estimated_months_to_esrd else 'unknown — act on current stage'}."
        )
    elif estimated_months_to_esrd and estimated_months_to_esrd <= 36:
        urgency = 'planned'
        priority_message = (
            f"PLANNED: eGFR {egfr:.0f} (Stage {stage}). "
            f"Estimated time to ESRD: ~{estimated_months_to_esrd} months. "
            f"Begin access planning now."
        )
    elif egfr < 30:
        urgency = 'prepare'
        priority_message = (
            f"PREPARE: eGFR {egfr:.0f} (Stage {stage}). "
            f"Begin dialysis education and access planning."
        )
    else:
        urgency = 'monitor'
        priority_message = f"MONITOR: eGFR {egfr:.0f} (Stage {stage}). Continue routine monitoring."

    # Compute target placement dates for each modality
    access_timing = {}
    for access_type, data in ACCESS_TYPES.items():
        total_lead = data['lead_time_months'] + data['maturation_time_months']
        if estimated_months_to_esrd:
            placement_months_from_now = max(0, estimated_months_to_esrd - total_lead)
            target_date = current_date + datetime.timedelta(days=int(placement_months_from_now * 30))
            status = 'overdue' if placement_months_from_now <= 0 else 'scheduled'
        else:
            target_date = None
            status = 'plan_needed' if urgency in ('urgent', 'planned') else 'not_yet_indicated'

        access_timing[access_type] = {
            'modality': data['label'],
            'target_placement_date': target_date.isoformat() if target_date else None,
            'lead_time_months': data['lead_time_months'],
            'maturation_months': data['maturation_time_months'],
            'total_time_months': total_lead,
            'status': status,
        }

    # Personalized recommendation
    prefers_home = patient_preferences.get('home_dialysis', False)
    if prefers_home:
        recommended_modality = 'pd_catheter'
        rationale = 'Patient preference for home-based peritoneal dialysis.'
    elif egfr < 15 and estimated_months_to_esrd and estimated_months_to_esrd < ACCESS_TYPES['av_fistula']['lead_time_months'] + ACCESS_TYPES['av_fistula']['maturation_time_months']:
        recommended_modality = 'av_graft'
        rationale = 'Insufficient time for AVF maturation; AVG offers faster access.'
    else:
        recommended_modality = 'av_fistula'
        rationale = 'AVF is the gold standard with best long-term outcomes (KDOQI guideline).'

    # Action plan
    if urgency == 'urgent':
        action_plan = [
            "Immediate referral to vascular surgery for access planning",
            "Order vein mapping ultrasound",
            f"Schedule {ACCESS_TYPES[recommended_modality]['label']} placement within 4 weeks",
            "Enroll patient in dialysis education program (HD vs PD choice)",
            "Evaluate for pre-emptive kidney transplant listing",
            "Social work referral for disability planning and transport",
        ]
    elif urgency == 'planned':
        action_plan = [
            "Refer to vascular surgery within 1 month",
            "Order vein mapping ultrasound",
            f"Plan {ACCESS_TYPES[recommended_modality]['label']} by {access_timing[recommended_modality].get('target_placement_date', 'TBD')}",
            "Dialysis education: modality selection discussion",
            "Hepatitis B vaccination if not immune",
            "Anemia optimization before access surgery",
        ]
    elif urgency == 'prepare':
        action_plan = [
            "Initiate dialysis education program",
            "Preserve arm veins — no IV lines or blood draws from non-dominant arm",
            "Discuss modality options (HD, PD, conservative management, transplant)",
            "Vascular surgery referral within 3 months",
            "Update advance directives",
        ]
    else:
        action_plan = [
            "Continue monitoring eGFR trajectory",
            "Preserve non-dominant arm veins (no IVs, no BP cuffs) when eGFR < 30",
            "Patient education on CKD stages and future planning",
        ]

    # Contraindications/considerations
    contraindications = []
    if patient_preferences.get('prior_abdominal_surgery'):
        contraindications.append("Prior abdominal surgery may complicate PD catheter insertion — assess with surgeon")
    if patient_preferences.get('poor_hand_function'):
        contraindications.append("Limited hand dexterity may impact home PD exchanges — evaluate with OT")

    return {
        'urgency': urgency,
        'priority_message': priority_message,
        'current_egfr': egfr,
        'stage': stage,
        'estimated_months_to_esrd': estimated_months_to_esrd,
        'recommended_modality': recommended_modality,
        'recommended_modality_label': ACCESS_TYPES[recommended_modality]['label'],
        'rationale': rationale,
        'access_timing': access_timing,
        'action_plan': action_plan,
        'contraindications': contraindications,
        'modality_options': ACCESS_TYPES,
        'guideline': 'KDOQI Clinical Practice Guidelines for Vascular Access 2019',
    }
