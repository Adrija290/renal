import numpy as np
import pandas as pd
from scipy import stats


def calculate_demographic_parity(predictions_df, protected_attrs=['age_group', 'race', 'gender']):
    metrics = {}
    for attr in protected_attrs:
        if attr in predictions_df.columns:
            groups = predictions_df[attr].unique()
            parity_diffs = []
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    group_i_pos = predictions_df[predictions_df[attr] == groups[i]]['positive_risk'].mean()
                    group_j_pos = predictions_df[predictions_df[attr] == groups[j]]['positive_risk'].mean()
                    parity_diffs.append(abs(group_i_pos - group_j_pos))
            metrics[attr] = {
                'max_parity_diff': max(parity_diffs) if parity_diffs else 0,
                'mean_parity_diff': np.mean(parity_diffs) if parity_diffs else 0
            }
    return metrics


def run_fairness_audit(start_date=None, end_date=None):
    return {
        'status': 'demo',
        'message': 'Fairness audit running in demo mode',
        'demographic_parity': {},
        'equalized_odds': {},
        'recommendations': ['No fairness violations detected in demo mode']
    }
