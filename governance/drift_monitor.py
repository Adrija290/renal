import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp
from sklearn.metrics import roc_auc_score
from database.db import db_session
from database.models import PredictionLog
from models.ensemble import EnsemblePredictor

class DriftMonitor:
    def __init__(self, reference_data_path='models_cache/reference_distributions.npz'):
        self.reference_dist = np.load(reference_data_path, allow_pickle=True) if exists(reference_data_path) else None
        self.detector = EnsemblePredictor()  # Reuse existing model for scoring
    
    def detect_data_drift(self, new_data_df, features=['age', 'creatinine', 'gfr', 'hemoglobin']):
        """Kolmogorov-Smirnov test for feature distribution drift"""
        drifts = {}
        p_values = {}
        
        for feature in features:
            if feature in new_data_df.columns and self.reference_dist is not None:
                ref_dist = self.reference_dist[f'ref_{feature}']
                stat, pval = ks_2samp(ref_dist, new_data_df[feature].values)
                drifts[feature] = {'ks_statistic': stat}
                p_values[feature] = pval
                
                if pval < 0.01:
                    drifts[feature]['alert'] = 'HIGH_DRIFT'
                elif pval < 0.05:
                    drifts[feature]['alert'] = 'MEDIUM_DRIFT'
                else:
                    drifts[feature]['alert'] = 'NO_DRIFT'
        
        return {'drifts': drifts, 'p_values': p_values}
    
    def detect_concept_drift(self, new_predictions_df):
        """Monitor prediction distribution changes (KL divergence)"""
        if len(new_predictions_df) < 100:
            return {'status': 'insufficient_data'}
        
        recent_probs = new_predictions_df['risk_score'].values
        if self.reference_dist is None:
            # First run - establish baseline
            np.savez('models_cache/reference_distributions.npz',
                    ref_risk=recent_probs,
                    ref_age=new_predictions_df['age'].values if 'age' in new_predictions_df else np.array([]),
                    ref_gfr=new_predictions_df['gfr'].values if 'gfr' in new_predictions_df else np.array([]))
            return {'status': 'baseline_established'}
        
        ref_probs = self.reference_dist['ref_risk']
        kl_div = entropy(ref_probs, recent_probs) if len(ref_probs) > 0 else 0
        
        alert = 'NO_DRIFT'
        if kl_div > 0.1:
            alert = 'HIGH_DRIFT - Retraining recommended'
        elif kl_div > 0.05:
            alert = 'MEDIUM_DRIFT - Monitor closely'
        
        return {
            'kl_divergence': float(kl_div),
            'alert_level': alert,
            'recommendation': 'Schedule retraining' if 'HIGH' in alert else 'No action needed'
        }
    
    def performance_drift(self, new_data_df, true_labels_col='actual_outcome'):
        """Monitor model performance degradation"""
        if true_labels_col not in new_data_df.columns:
            return {'status': 'no_ground_truth'}
        
        y_true = new_data_df[true_labels_col]
        y_pred = new_data_df['risk_score']
        
        auc = roc_auc_score(y_true, y_pred)
        return {
            'current_auc': float(auc),
            'performance_alert': 'LOW_PERFORMANCE' if auc < 0.75 else 'NORMAL'
        }

def run_full_drift_monitor(new_data_file='data/recent_predictions.csv'):
    """Run comprehensive drift monitoring"""
    df = pd.read_csv(new_data_file)
    monitor = DriftMonitor()
    
    data_drift = monitor.detect_data_drift(df)
    concept_drift = monitor.detect_concept_drift(df)
    perf_drift = monitor.performance_drift(df)
    
    return {
        'data_drift': data_drift,
        'concept_drift': concept_drift,
        'performance_drift': perf_drift,
        'overall_status': 'ALERT' if any('HIGH' in str(v) for d in [data_drift, concept_drift, perf_drift] for v in d.values()) else 'MONITORED'
    }

def exists(path):
    import os
    return os.path.exists(path)

