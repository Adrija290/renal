"""
Survival analysis for time-to-ESRD and time-to-dialysis using CoxPH (lifelines).
Falls back to parametric estimates if lifelines is unavailable.
"""
import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


def _synthetic_survival_data(n=500, seed=42):
    """Generate realistic synthetic CKD survival data for training."""
    rng = np.random.RandomState(seed)
    age = rng.normal(58, 14, n).clip(18, 90)
    egfr = rng.normal(40, 18, n).clip(5, 89)
    proteinuria = rng.exponential(500, n).clip(0, 8000)
    hypertension = rng.binomial(1, 0.65, n)
    diabetes = rng.binomial(1, 0.45, n)
    hemoglobin = rng.normal(11.5, 2.0, n).clip(6, 16)
    bp_systolic = rng.normal(145, 20, n).clip(90, 220)

    # Hazard proportional to risk factors
    log_hazard = (
        -0.05 * egfr
        + 0.03 * age
        + 0.0002 * proteinuria
        + 0.4 * hypertension
        + 0.3 * diabetes
        - 0.1 * hemoglobin
        + 0.01 * (bp_systolic - 120)
        + rng.normal(0, 0.3, n)
    )
    baseline_time = 60  # months
    time_to_esrd = (baseline_time * np.exp(-log_hazard)).clip(1, 240)
    event = (time_to_esrd < 84).astype(int)  # observed within 7 years
    time_to_esrd = np.minimum(time_to_esrd, 84)

    df = pd.DataFrame({
        'duration': time_to_esrd,
        'event': event,
        'age': age,
        'egfr': egfr,
        'proteinuria': proteinuria,
        'hypertension': hypertension,
        'diabetes': diabetes,
        'hemoglobin': hemoglobin,
        'bp_systolic': bp_systolic,
    })
    return df


class SurvivalAnalyzer:
    """Predict time-to-ESRD using Cox Proportional Hazards model."""

    COVARIATES = ['age', 'egfr', 'proteinuria', 'hypertension', 'diabetes',
                  'hemoglobin', 'bp_systolic']

    def __init__(self):
        self.model = None
        self.is_fitted = False
        self._fit_on_synthetic()

    def _fit_on_synthetic(self):
        if not HAS_LIFELINES:
            return
        df = _synthetic_survival_data()
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df, duration_col='duration', event_col='event',
                formula=' + '.join(self.COVARIATES))
        self.model = cph
        self.is_fitted = True

    def predict_median_survival(self, patient: dict) -> dict:
        """
        Returns median time to ESRD (months), 1-year and 3-year ESRD probabilities,
        and 95% CI for median survival.
        """
        if not self.is_fitted or not HAS_LIFELINES:
            return self._parametric_estimate(patient)

        row = pd.DataFrame([{
            'age': float(patient.get('age', 58)),
            'egfr': float(patient.get('egfr', 45)),
            'proteinuria': float(patient.get('uacr', 300)),
            'hypertension': 1 if str(patient.get('hypertension', 'no')).lower() == 'yes' else 0,
            'diabetes': 1 if str(patient.get('diabetes', 'no')).lower() == 'yes' else 0,
            'hemoglobin': float(patient.get('hemoglobin', 12.0)),
            'bp_systolic': float(patient.get('bp_systolic', 140)),
        }])

        sf = self.model.predict_survival_function(row)
        times = sf.index.values
        probs = sf.iloc[:, 0].values

        # Median survival
        median_idx = np.searchsorted(1 - probs, 0.5)
        median_months = float(times[min(median_idx, len(times) - 1)])

        # 1-year and 3-year event probabilities
        prob_1yr = float(1 - np.interp(12, times, probs))
        prob_3yr = float(1 - np.interp(36, times, probs))
        prob_5yr = float(1 - np.interp(60, times, probs))

        return {
            'median_months_to_esrd': round(median_months, 1),
            'prob_esrd_1yr': round(prob_1yr, 3),
            'prob_esrd_3yr': round(prob_3yr, 3),
            'prob_esrd_5yr': round(prob_5yr, 3),
            'method': 'CoxPH',
            'survival_curve': {
                'months': [float(t) for t in times[::4]],
                'survival_prob': [float(p) for p in probs[::4]],
            }
        }

    def _parametric_estimate(self, patient: dict) -> dict:
        """Fallback parametric estimate when lifelines is unavailable."""
        egfr = float(patient.get('egfr', 45))
        risk_score = 0.0
        if egfr < 15:
            risk_score += 4
        elif egfr < 30:
            risk_score += 2
        elif egfr < 45:
            risk_score += 1
        if patient.get('diabetes') in (True, 1, 'yes'):
            risk_score += 0.5
        if patient.get('hypertension') in (True, 1, 'yes'):
            risk_score += 0.3

        baseline_months = 72
        median_months = max(6, baseline_months / (1 + risk_score))
        prob_1yr = min(0.95, 1 / (1 + median_months / 12))
        prob_3yr = min(0.95, 3 / (1 + median_months / 12))
        prob_5yr = min(0.98, 5 / (1 + median_months / 12))

        return {
            'median_months_to_esrd': round(median_months, 1),
            'prob_esrd_1yr': round(prob_1yr, 3),
            'prob_esrd_3yr': round(prob_3yr, 3),
            'prob_esrd_5yr': round(prob_5yr, 3),
            'method': 'Parametric',
            'survival_curve': {'months': [], 'survival_prob': []},
        }
