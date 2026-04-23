"""
Wearable device data integration.
Supports blood pressure monitors, continuous glucose monitors (CGM), and fitness trackers.
Normalizes data and stores wearable readings for trend analysis.
"""
import datetime
import numpy as np
import random


SUPPORTED_DEVICES = {
    'bp_monitor': {
        'name': 'Blood Pressure Monitor',
        'fields': ['systolic_bp', 'diastolic_bp', 'heart_rate'],
        'brands': ['Omron', 'Withings', 'iHealth', 'QardioArm'],
        'frequency': 'Multiple times daily',
    },
    'cgm': {
        'name': 'Continuous Glucose Monitor',
        'fields': ['blood_glucose'],
        'brands': ['Dexcom G7', 'Abbott FreeStyle Libre', 'Medtronic Guardian'],
        'frequency': 'Every 5 minutes',
    },
    'smart_scale': {
        'name': 'Smart Scale',
        'fields': ['weight_kg'],
        'brands': ['Withings Body+', 'Nokia', 'Fitbit Aria'],
        'frequency': 'Daily',
    },
    'fitness_tracker': {
        'name': 'Fitness Tracker',
        'fields': ['steps', 'heart_rate'],
        'brands': ['Fitbit', 'Apple Watch', 'Garmin', 'Samsung Galaxy Watch'],
        'frequency': 'Continuous',
    },
}


def process_bp_reading(raw_data: dict) -> dict:
    """Validate and normalize a blood pressure reading."""
    systolic = float(raw_data.get('systolic', raw_data.get('sys', 0)))
    diastolic = float(raw_data.get('diastolic', raw_data.get('dia', 0)))
    heart_rate = float(raw_data.get('heart_rate', raw_data.get('pulse', 0)))
    timestamp = raw_data.get('timestamp', datetime.datetime.utcnow().isoformat())

    alerts = []
    if systolic >= 180 or diastolic >= 120:
        alerts.append({'severity': 'critical', 'message': f'Hypertensive crisis: {systolic}/{diastolic} mmHg — seek immediate medical attention'})
    elif systolic >= 140 or diastolic >= 90:
        alerts.append({'severity': 'high', 'message': f'Hypertension Stage 2: {systolic}/{diastolic} mmHg — contact your care team'})
    elif systolic >= 130:
        alerts.append({'severity': 'moderate', 'message': f'Elevated BP: {systolic}/{diastolic} mmHg — take medications as prescribed'})

    if systolic < 90 or diastolic < 60:
        alerts.append({'severity': 'high', 'message': f'Low BP: {systolic}/{diastolic} mmHg — sit down and call your doctor if symptomatic'})

    bp_category = _classify_bp(systolic, diastolic)

    return {
        'device_type': 'bp_monitor',
        'timestamp': timestamp,
        'systolic_bp': systolic,
        'diastolic_bp': diastolic,
        'heart_rate': heart_rate,
        'bp_category': bp_category,
        'alerts': alerts,
        'valid': systolic > 50 and systolic < 300 and diastolic > 30 and diastolic < 200,
    }


def _classify_bp(systolic: float, diastolic: float) -> str:
    if systolic < 120 and diastolic < 80:
        return 'Normal'
    if systolic < 130 and diastolic < 80:
        return 'Elevated'
    if systolic < 140 or diastolic < 90:
        return 'Stage 1 Hypertension'
    if systolic >= 140 or diastolic >= 90:
        return 'Stage 2 Hypertension'
    return 'Unknown'


def process_cgm_reading(raw_data: dict) -> dict:
    """Validate and normalize a CGM glucose reading."""
    glucose = float(raw_data.get('glucose', raw_data.get('value', 100)))
    unit = raw_data.get('unit', 'mg/dL')
    timestamp = raw_data.get('timestamp', datetime.datetime.utcnow().isoformat())

    if unit in ('mmol/L', 'mmol'):
        glucose = glucose * 18.0

    alerts = []
    if glucose >= 400:
        alerts.append({'severity': 'critical', 'message': f'Critically high glucose: {glucose:.0f} mg/dL — check for DKA, contact care team'})
    elif glucose >= 250:
        alerts.append({'severity': 'high', 'message': f'Very high glucose: {glucose:.0f} mg/dL — administer correction if on insulin'})
    elif glucose >= 180:
        alerts.append({'severity': 'moderate', 'message': f'Elevated glucose: {glucose:.0f} mg/dL — review meal choices and medications'})
    elif glucose < 70:
        alerts.append({'severity': 'critical', 'message': f'Hypoglycemia: {glucose:.0f} mg/dL — consume 15g fast-acting carbohydrates immediately'})
    elif glucose < 80:
        alerts.append({'severity': 'high', 'message': f'Low glucose trending: {glucose:.0f} mg/dL — have a snack ready'})

    return {
        'device_type': 'cgm',
        'timestamp': timestamp,
        'blood_glucose': glucose,
        'unit': 'mg/dL',
        'glucose_category': _classify_glucose(glucose),
        'alerts': alerts,
        'valid': 20 < glucose < 600,
    }


def _classify_glucose(glucose: float) -> str:
    if glucose < 70:
        return 'Hypoglycemia'
    if glucose < 100:
        return 'Normal (fasting)'
    if glucose < 140:
        return 'Normal (post-meal)'
    if glucose < 180:
        return 'Mildly Elevated'
    return 'Hyperglycemia'


def process_weight_reading(raw_data: dict) -> dict:
    """Validate weight reading and detect fluid overload."""
    weight_kg = float(raw_data.get('weight_kg', raw_data.get('weight', 70)))
    prev_weight_kg = raw_data.get('previous_weight_kg')
    timestamp = raw_data.get('timestamp', datetime.datetime.utcnow().isoformat())

    alerts = []
    if prev_weight_kg:
        delta = weight_kg - float(prev_weight_kg)
        if delta >= 2.0:
            alerts.append({
                'severity': 'high',
                'message': f'Weight gain of {delta:.1f}kg since last reading — may indicate fluid retention. Contact your care team.',
            })
        elif delta >= 1.0:
            alerts.append({
                'severity': 'moderate',
                'message': f'Weight gain of {delta:.1f}kg — monitor fluid intake and restrict sodium.',
            })

    return {
        'device_type': 'smart_scale',
        'timestamp': timestamp,
        'weight_kg': weight_kg,
        'weight_delta_kg': (weight_kg - float(prev_weight_kg)) if prev_weight_kg else None,
        'alerts': alerts,
        'valid': 20 < weight_kg < 300,
    }


def generate_demo_wearable_data(patient_id: str, days: int = 30,
                                  systolic_base: int = 145,
                                  glucose_base: float = 165.0) -> list:
    """Generate synthetic wearable readings for demo purposes."""
    rng = random.Random(hash(patient_id) % 1000)
    readings = []
    now = datetime.datetime.utcnow()

    for day in range(days):
        date = now - datetime.timedelta(days=days - day)
        for _ in range(2):
            readings.append({
                'device_type': 'bp_monitor',
                'patient_id': patient_id,
                'timestamp': date.isoformat(),
                'systolic_bp': systolic_base + rng.randint(-15, 15),
                'diastolic_bp': rng.randint(75, 95),
                'heart_rate': rng.randint(62, 85),
            })
        readings.append({
            'device_type': 'cgm',
            'patient_id': patient_id,
            'timestamp': date.isoformat(),
            'blood_glucose': max(80, glucose_base + rng.uniform(-40, 40)),
        })
        readings.append({
            'device_type': 'smart_scale',
            'patient_id': patient_id,
            'timestamp': date.isoformat(),
            'weight_kg': 78.0 + rng.uniform(-0.5, 0.5),
        })

    return readings


def compute_wearable_trends(readings: list) -> dict:
    """Compute summary statistics and trends from wearable data."""
    bp_readings = [r for r in readings if r['device_type'] == 'bp_monitor']
    glucose_readings = [r for r in readings if r['device_type'] == 'cgm']
    weight_readings = [r for r in readings if r['device_type'] == 'smart_scale']

    result = {}

    if bp_readings:
        systolics = [r['systolic_bp'] for r in bp_readings if 'systolic_bp' in r]
        if systolics:
            result['bp'] = {
                'mean_systolic': round(np.mean(systolics), 1),
                'max_systolic': max(systolics),
                'min_systolic': min(systolics),
                'hypertension_readings_pct': round(sum(1 for s in systolics if s >= 140) / len(systolics) * 100, 1),
                'recent_systolics': systolics[-7:],
            }

    if glucose_readings:
        glucoses = [r['blood_glucose'] for r in glucose_readings if 'blood_glucose' in r]
        if glucoses:
            result['glucose'] = {
                'mean_glucose': round(np.mean(glucoses), 1),
                'time_in_range_pct': round(sum(1 for g in glucoses if 80 <= g <= 180) / len(glucoses) * 100, 1),
                'hypoglycemia_events': sum(1 for g in glucoses if g < 70),
                'hyperglycemia_events': sum(1 for g in glucoses if g > 250),
                'recent_glucoses': glucoses[-7:],
            }

    if weight_readings:
        weights = [r['weight_kg'] for r in weight_readings if 'weight_kg' in r]
        if weights:
            result['weight'] = {
                'current_kg': weights[-1],
                'change_7d': round(weights[-1] - weights[max(0, len(weights) - 7)], 2) if len(weights) > 1 else 0,
                'recent_weights': weights[-7:],
            }

    return result
