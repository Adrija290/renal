import numpy as np
import pandas as pd
from scipy import stats
from database.db import db_session
from database.models import PredictionLog

def calculate_demographic_parity(predictions_df, protected_attrs=['age_group', 'race', 'gender']):
    """
    Calculate demographic parity difference across protected attributes.
    Lower values indicate fairer models (closer to 0).
    """
    metrics = {}
    for attr in protected_attrs:
        if attr in predictions_df.columns:
            groups = predictions_df[attr].unique()
            parity_diffs = []
            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    group_i_pos = predictions_df[predictions_df[attr] == groups[i]]['positive_risk'].mean()
                    group_j_pos = predictions_df[predictions_df[attr] == groups[j]]['positive_risk'].mean()
                    parity_diffs.append(abs(group_i_pos - group_j_pos))
            metrics[attr] = {
                'max_parity_diff': max(parity_diffs) if parity_diffs else 0,
                'mean_parity_diff': np.mean(parity_diffs) if parity_diffs else 0
            }
    return metrics

def calculate_equalized_odds(predictions_df, protected_attrs=['race']):
    """Equalized odds: Equal TPR/FPR across groups"""
    metrics = {}
    for attr in protected_attrs:
        if attr in predictions_df.columns:
            groups = predictions_df[attr].unique()
            tpr = {}, fpr = {}
            for group in groups:
                group_data = predictions_df[predictions_df[attr] == group]
                tp = ((group_data['positive_risk'] > 0.7) & (group_data['actual_outcome'] == 1)).sum()
                fp = ((group_data['positive_risk'] > 0.7) & (group_data['actual_outcome'] == 0)).sum()
                fn = ((group_data['positive_risk'] <= 0.7) & (group_data['actual_outcome'] == 1)).sum()
                tn = ((group_data['positive_risk'] <= 0.7) & (group_data['actual_outcome'] == 0)).sum()
                
                tpr[group] = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr[group] = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            metrics[attr] = {
                'tpr_diff': max(tpr.values()) - min(tpr.values()),
                'fpr_diff': max(fpr.values()) - min(fpr.values())
            }
    return metrics

def run_fairness_audit(start_date=None, end_date=None):
    """Comprehensive fairness audit"""
    logs = db_session.query(PredictionLog).filter(
        PredictionLog.created_at >= start_date,
        PredictionLog.created_at <= end_date
    ).all()
    
    if not logs:
        return {'status': 'no_data', 'message': 'No predictions found for audit period'}
    
    df = pd.DataFrame([log.to_dict() for log in logs])
    parity_metrics = calculate_demographic_parity(df)
    odds_metrics = calculate_equalized_odds(df)
    
    return {
        'status': 'complete',
        'demographic_parity': parity_metrics,
        'equalized_odds': odds_metrics,
        'recommendations': _generate_fairness_recommendations(parity_metrics, odds_metrics)
    }

def _generate_fairness_recommendations(parity, odds):
    recs = []
    if any(m['max_parity_diff'] > 0.1 for m in parity.values()):
        recs.append('High demographic parity violation - consider reweighting training data')
    if any(m['tpr_diff'] > 0.1 or m['fpr_diff'] > 0.1 for m in odds.values()):
        recs.append('Equalized odds violation - adversarial debiasing recommended')
    return recs if recs else ['Model appears fair across protected groups']

