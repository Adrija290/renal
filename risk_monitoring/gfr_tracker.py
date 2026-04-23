"""
Real-time GFR stage tracking: monitors trajectory, predicts stage transitions,
and generates alerts on rapid decline or anomalous values.
"""
import datetime
import numpy as np
from models.gfr_forecaster import GFRForecaster, classify_ckd_stage, egfr_from_creatinine
from config import Config

STAGE_DESCRIPTIONS = {
    'G1': 'Normal or High kidney function (eGFR ≥90)',
    'G2': 'Mildly Decreased (eGFR 60-89)',
    'G3a': 'Mildly to Moderately Decreased (eGFR 45-59)',
    'G3b': 'Moderately to Severely Decreased (eGFR 30-44)',
    'G4': 'Severely Decreased (eGFR 15-29)',
    'G5': 'Kidney Failure (eGFR <15)',
}

STAGE_COLORS = {
    'G1': '#28a745', 'G2': '#5cb85c', 'G3a': '#ffc107',
    'G3b': '#fd7e14', 'G4': '#dc3545', 'G5': '#6f0000',
}

MONITORING_FREQUENCY = {
    'G1': '12 months', 'G2': '12 months', 'G3a': '6 months',
    'G3b': '6 months', 'G4': '3 months', 'G5': '1-3 months',
}


def analyze_patient_gfr(patient_id: str, lab_results: list) -> dict:
    """
    Analyze a patient's GFR history and return complete trajectory analysis.
    lab_results: list of LabResult model objects (ordered by date ascending).
    """
    if not lab_results:
        return {'error': 'No lab results available'}

    egfr_values = [lr.egfr for lr in lab_results if lr.egfr is not None]
    dates = [lr.date for lr in lab_results if lr.egfr is not None]

    if not egfr_values:
        return {'error': 'No eGFR values in lab results'}

    forecaster = GFRForecaster(horizon_months=36)
    forecast = forecaster.forecast(egfr_values)

    current_egfr = egfr_values[-1]
    current_stage = classify_ckd_stage(current_egfr)

    # Compute decline trend
    alerts = []
    if forecast.get('rapid_decline_alert'):
        alerts.append({
            'type': 'rapid_decline',
            'severity': 'critical',
            'message': (f"Rapid GFR decline detected: "
                        f"{abs(forecast['annual_decline_rate']):.1f} mL/min/1.73m²/year "
                        f"(threshold: {Config.GFR_RAPID_DECLINE_THRESHOLD}). "
                        f"Urgent nephrologist review recommended."),
        })

    # Stage worsening since last visit
    if len(egfr_values) >= 2:
        prev_stage = classify_ckd_stage(egfr_values[-2])
        if prev_stage != current_stage:
            stage_worse = _is_stage_worse(prev_stage, current_stage)
            if stage_worse:
                alerts.append({
                    'type': 'stage_progression',
                    'severity': 'high',
                    'message': (f"CKD stage worsened from {prev_stage} to {current_stage}. "
                                f"Increase monitoring frequency and review treatment plan."),
                })

    # Upcoming ESRD alert
    esrd_months = forecast.get('estimated_months_to_esrd')
    if esrd_months and esrd_months <= 18:
        alerts.append({
            'type': 'esrd_approaching',
            'severity': 'critical',
            'message': (f"ESRD predicted in approximately {esrd_months} months. "
                        f"Begin dialysis access preparation immediately. "
                        f"Refer to vascular surgery for AV fistula planning."),
        })
    elif esrd_months and esrd_months <= 36:
        alerts.append({
            'type': 'esrd_approaching',
            'severity': 'high',
            'message': (f"ESRD predicted in approximately {esrd_months} months. "
                        f"Begin planning for renal replacement therapy. "
                        f"Educate patient on dialysis options."),
        })

    # Historical chart data for frontend
    history_labels = [d.strftime('%b %Y') if d else '' for d in dates]
    forecast_labels = [
        f"+{(i + 1) * 3}mo" for i in range(len(forecast['forecasted_egfr']))
    ]

    return {
        'patient_id': patient_id,
        'current_egfr': round(current_egfr, 1),
        'current_stage': current_stage,
        'stage_description': STAGE_DESCRIPTIONS.get(current_stage, ''),
        'stage_color': STAGE_COLORS.get(current_stage, '#6c757d'),
        'monitoring_frequency': MONITORING_FREQUENCY.get(current_stage, ''),
        'annual_decline_rate': forecast.get('annual_decline_rate'),
        'forecast': forecast,
        'alerts': alerts,
        'chart': {
            'history_labels': history_labels,
            'history_values': egfr_values,
            'forecast_labels': forecast_labels,
            'forecast_values': forecast['forecasted_egfr'],
            'ci_lower': forecast['ci_lower'],
            'ci_upper': forecast['ci_upper'],
        },
        'stage_transitions': forecast.get('stage_transitions', []),
    }


def _is_stage_worse(from_stage: str, to_stage: str) -> bool:
    order = ['G1', 'G2', 'G3a', 'G3b', 'G4', 'G5']
    try:
        return order.index(to_stage) > order.index(from_stage)
    except ValueError:
        return False
