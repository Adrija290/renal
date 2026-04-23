"""
GFR trajectory forecasting using ARIMA / linear trend extrapolation.
Predicts future eGFR values, stage transitions, and alerts on rapid decline.
"""
import numpy as np
import pandas as pd
from config import Config

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def egfr_from_creatinine(creatinine: float, age: int, sex: str, race: str = 'Other') -> float:
    """CKD-EPI 2021 eGFR formula (race-free version)."""
    if sex.upper() in ('F', 'FEMALE'):
        kappa, alpha = 0.7, -0.241
        sex_factor = 1.012
    else:
        kappa, alpha = 0.9, -0.302
        sex_factor = 1.0

    ratio = creatinine / kappa
    if ratio < 1:
        gfr = 142 * (ratio ** alpha) * (0.9938 ** age) * sex_factor
    else:
        gfr = 142 * (ratio ** -1.200) * (0.9938 ** age) * sex_factor

    return round(max(1.0, gfr), 1)


def classify_ckd_stage(egfr: float) -> str:
    for stage, (low, high) in Config.CKD_STAGES.items():
        if low <= egfr < high:
            return stage
    return 'G5'


class GFRForecaster:
    """Forecast GFR trajectory and predict CKD stage transitions."""

    def __init__(self, horizon_months: int = 36):
        self.horizon_months = horizon_months

    def forecast(self, historical_egfr: list[float],
                  historical_dates: list = None) -> dict:
        """
        historical_egfr: list of eGFR values in chronological order (3-month intervals assumed)
        Returns forecast dict with predictions, stage transitions, and alerts.
        """
        n = len(historical_egfr)
        if n < 2:
            return self._insufficient_data(historical_egfr)

        values = np.array(historical_egfr, dtype=float)
        forecast_points = self.horizon_months // 3

        if n >= 6 and HAS_STATSMODELS:
            try:
                model = ARIMA(values, order=(1, 1, 1))
                fit = model.fit()
                forecast_result = fit.get_forecast(steps=forecast_points)
                forecasted = forecast_result.predicted_mean.tolist()
                ci_lower = forecast_result.conf_int()['lower y'].tolist()
                ci_upper = forecast_result.conf_int()['upper y'].tolist()
            except Exception:
                forecasted, ci_lower, ci_upper = self._linear_forecast(values, forecast_points)
        else:
            forecasted, ci_lower, ci_upper = self._linear_forecast(values, forecast_points)

        forecasted = [max(1.0, v) for v in forecasted]
        ci_lower = [max(1.0, v) for v in ci_lower]
        ci_upper = [min(130.0, v) for v in ci_upper]

        # Annual decline rate (mL/min/1.73m²/year)
        if n >= 4:
            slope = np.polyfit(range(n), values, 1)[0]
            annual_decline = slope * 4
        else:
            annual_decline = (values[-1] - values[0]) / max(1, n - 1) * 4

        current_stage = classify_ckd_stage(float(values[-1]))
        current_egfr = float(values[-1])

        # Find stage transition points
        transitions = []
        all_forecasted = list(values) + forecasted
        stages_over_time = [classify_ckd_stage(v) for v in all_forecasted]
        for i in range(len(stages_over_time) - 1):
            if stages_over_time[i] != stages_over_time[i + 1]:
                month_offset = (i - n + 1) * 3
                transitions.append({
                    'from_stage': stages_over_time[i],
                    'to_stage': stages_over_time[i + 1],
                    'estimated_months': month_offset,
                })

        # Predict when ESRD (eGFR < 15) will be reached
        esrd_months = None
        for i, val in enumerate(forecasted):
            if val < 15:
                esrd_months = (i + 1) * 3
                break

        # Rapid decline alert
        rapid_decline = annual_decline < -Config.GFR_RAPID_DECLINE_THRESHOLD

        return {
            'current_egfr': current_egfr,
            'current_stage': current_stage,
            'annual_decline_rate': round(annual_decline, 2),
            'rapid_decline_alert': rapid_decline,
            'forecasted_egfr': [round(v, 1) for v in forecasted],
            'ci_lower': [round(v, 1) for v in ci_lower],
            'ci_upper': [round(v, 1) for v in ci_upper],
            'stage_transitions': transitions,
            'estimated_months_to_esrd': esrd_months,
            'forecast_horizon_months': self.horizon_months,
            'historical_egfr': list(historical_egfr),
        }

    def _linear_forecast(self, values: np.ndarray, steps: int):
        n = len(values)
        x = np.arange(n)
        slope, intercept = np.polyfit(x, values, 1)
        future_x = np.arange(n, n + steps)
        forecasted = (slope * future_x + intercept).tolist()
        residuals = values - (slope * x + intercept)
        std_err = np.std(residuals) * 1.96
        ci_lower = [v - std_err for v in forecasted]
        ci_upper = [v + std_err for v in forecasted]
        return forecasted, ci_lower, ci_upper

    def _insufficient_data(self, historical_egfr: list) -> dict:
        current = float(historical_egfr[-1]) if historical_egfr else 45.0
        return {
            'current_egfr': current,
            'current_stage': classify_ckd_stage(current),
            'annual_decline_rate': None,
            'rapid_decline_alert': False,
            'forecasted_egfr': [],
            'ci_lower': [],
            'ci_upper': [],
            'stage_transitions': [],
            'estimated_months_to_esrd': None,
            'forecast_horizon_months': self.horizon_months,
            'historical_egfr': list(historical_egfr),
            'message': 'Insufficient data for forecasting (need ≥2 measurements)',
        }
