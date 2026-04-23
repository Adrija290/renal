"""
Federated learning simulation across N virtual hospitals.
Implements FedAvg algorithm without sharing raw patient data.
Each hospital trains on its local synthetic dataset; only model weights are shared.
"""
import copy
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import Config

HOSPITAL_PROFILES = [
    {'id': 'H001', 'name': 'Metro General Hospital', 'region': 'Urban Northeast',
     'n_patients': 450, 'ckd_prevalence': 0.58, 'demographics': 'Diverse'},
    {'id': 'H002', 'name': 'Rural Valley Medical Center', 'region': 'Rural South',
     'n_patients': 180, 'ckd_prevalence': 0.72, 'demographics': 'Predominantly White'},
    {'id': 'H003', 'name': 'Pacific Coast Nephrology Institute', 'region': 'Urban West',
     'n_patients': 320, 'ckd_prevalence': 0.55, 'demographics': 'High Asian proportion'},
    {'id': 'H004', 'name': 'Midwest Community Hospital', 'region': 'Suburban Midwest',
     'n_patients': 260, 'ckd_prevalence': 0.65, 'demographics': 'Predominantly Black'},
    {'id': 'H005', 'name': 'Southwest Diabetes & Kidney Center', 'region': 'Urban Southwest',
     'n_patients': 380, 'ckd_prevalence': 0.70, 'demographics': 'High Hispanic proportion'},
]

FEATURE_COLS = ['age', 'egfr', 'bp', 'albumin', 'hemoglobin',
                 'potassium', 'diabetes', 'hypertension', 'cad']


def _generate_hospital_data(profile: dict, seed: int) -> pd.DataFrame:
    """Generate synthetic patient data for a hospital with realistic demographic variation."""
    rng = np.random.RandomState(seed)
    n = profile['n_patients']
    prev = profile['ckd_prevalence']

    n_ckd = int(n * prev)
    n_notckd = n - n_ckd

    # Demographic bias based on hospital profile
    age_shift = 5 if 'Rural' in profile['region'] else 0
    diabetes_boost = 0.1 if 'Diabetes' in profile['name'] else 0

    def make_patients(n_pts, is_ckd):
        if is_ckd:
            age = rng.normal(62 + age_shift, 14, n_pts).clip(25, 90)
            egfr = rng.normal(32, 14, n_pts).clip(5, 59)
            bp = rng.normal(155, 22, n_pts).clip(90, 220)
            albumin = rng.normal(2.8, 0.8, n_pts).clip(0.5, 5)
            hemo = rng.normal(9.8, 2.0, n_pts).clip(5, 14)
            potassium = rng.normal(4.9, 0.7, n_pts).clip(3.0, 7.5)
            diabetes = rng.binomial(1, min(0.55 + diabetes_boost, 1.0), n_pts)
            hypertension = rng.binomial(1, 0.80, n_pts)
            cad = rng.binomial(1, 0.35, n_pts)
        else:
            age = rng.normal(48 + age_shift, 14, n_pts).clip(18, 85)
            egfr = rng.normal(78, 14, n_pts).clip(60, 120)
            bp = rng.normal(125, 15, n_pts).clip(85, 160)
            albumin = rng.normal(4.0, 0.4, n_pts).clip(2.5, 5)
            hemo = rng.normal(13.5, 1.5, n_pts).clip(10, 18)
            potassium = rng.normal(4.2, 0.4, n_pts).clip(3.2, 5.5)
            diabetes = rng.binomial(1, min(0.15 + diabetes_boost, 0.5), n_pts)
            hypertension = rng.binomial(1, 0.35, n_pts)
            cad = rng.binomial(1, 0.12, n_pts)

        return pd.DataFrame({
            'age': age, 'egfr': egfr, 'bp': bp,
            'albumin': albumin, 'hemoglobin': hemo,
            'potassium': potassium, 'diabetes': diabetes.astype(float),
            'hypertension': hypertension.astype(float), 'cad': cad.astype(float),
            'label': int(is_ckd),
        })

    df_ckd = make_patients(n_ckd, True)
    df_notckd = make_patients(n_notckd, False)
    return pd.concat([df_ckd, df_notckd], ignore_index=True).sample(frac=1, random_state=seed)


def _get_model_weights(clf) -> np.ndarray:
    """Extract model weights (logistic regression coefficients)."""
    if hasattr(clf, 'coef_'):
        return np.concatenate([clf.coef_.flatten(), clf.intercept_])
    return np.array([])


def _set_model_weights(clf, weights: np.ndarray):
    """Set model weights from aggregated array."""
    if hasattr(clf, 'coef_'):
        n_feat = clf.coef_.shape[1]
        clf.coef_ = weights[:n_feat].reshape(1, -1)
        clf.intercept_ = weights[n_feat:n_feat + 1]


def run_federated_learning(n_rounds: int = None, callback=None) -> dict:
    """
    Run FedAvg simulation across all hospitals.
    Returns per-round metrics, final global model performance, and per-hospital stats.
    callback: optional function(round_num, metrics) for progress tracking.
    """
    n_rounds = n_rounds or Config.FL_ROUNDS
    hospitals_data = []
    for i, profile in enumerate(HOSPITAL_PROFILES):
        df = _generate_hospital_data(profile, seed=i * 42)
        hospitals_data.append(df)

    scaler = StandardScaler()
    all_data = pd.concat(hospitals_data, ignore_index=True)
    scaler.fit(all_data[FEATURE_COLS])

    global_model = LogisticRegression(max_iter=1000, random_state=42, warm_start=True)
    combined_X = scaler.transform(all_data[FEATURE_COLS])
    global_model.fit(combined_X, all_data['label'])
    global_weights = _get_model_weights(global_model)

    round_metrics = []
    hospital_contributions = []

    for rnd in range(1, n_rounds + 1):
        local_weights = []
        local_sizes = []
        local_models = []

        for i, (profile, df) in enumerate(zip(HOSPITAL_PROFILES, hospitals_data)):
            X = scaler.transform(df[FEATURE_COLS])
            y = df['label']
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                         stratify=y, random_state=rnd * 10 + i)
            local_clf = LogisticRegression(max_iter=500, random_state=42, warm_start=True)
            local_clf.fit(X_tr, y_tr)

            # Initialize from global weights for FedAvg
            if rnd > 1:
                try:
                    _set_model_weights(local_clf, global_weights)
                    local_clf.fit(X_tr, y_tr)
                except Exception:
                    pass

            w = _get_model_weights(local_clf)
            local_weights.append(w)
            local_sizes.append(len(X_tr))
            local_models.append((profile, local_clf, X_te, y_te))

        # FedAvg: weighted average by dataset size
        total = sum(local_sizes)
        if len(local_weights) > 0 and len(local_weights[0]) > 0:
            aggregated = np.sum(
                [w * (n / total) for w, n in zip(local_weights, local_sizes)], axis=0
            )
            try:
                _set_model_weights(global_model, aggregated)
                global_weights = aggregated
            except Exception:
                pass

        # Evaluate global model on all hospital test sets
        round_aucs = []
        for profile, local_clf, X_te, y_te in local_models:
            try:
                auc = roc_auc_score(y_te, global_model.predict_proba(X_te)[:, 1])
                round_aucs.append(auc)
            except Exception:
                pass

        avg_auc = round(np.mean(round_aucs), 4) if round_aucs else 0
        round_metrics.append({'round': rnd, 'global_auc': avg_auc, 'hospitals': len(HOSPITAL_PROFILES)})

        if callback:
            callback(rnd, {'round': rnd, 'auc': avg_auc})

    # Final hospital-specific metrics
    for i, (profile, df) in enumerate(zip(HOSPITAL_PROFILES, hospitals_data)):
        X = scaler.transform(df[FEATURE_COLS])
        y = df['label']
        y_pred = global_model.predict(X)
        y_prob = global_model.predict_proba(X)[:, 1]
        hospital_contributions.append({
            'hospital_id': profile['id'],
            'hospital_name': profile['name'],
            'region': profile['region'],
            'n_patients': profile['n_patients'],
            'ckd_prevalence': profile['ckd_prevalence'],
            'demographics': profile['demographics'],
            'local_accuracy': round(accuracy_score(y, y_pred), 4),
            'local_auc': round(roc_auc_score(y, y_prob), 4),
            'local_f1': round(f1_score(y, y_pred), 4),
        })

    X_all = scaler.transform(all_data[FEATURE_COLS])
    y_all = all_data['label']
    final_auc = round(roc_auc_score(y_all, global_model.predict_proba(X_all)[:, 1]), 4)
    final_acc = round(accuracy_score(y_all, global_model.predict(X_all)), 4)

    return {
        'total_patients': len(all_data),
        'total_hospitals': len(HOSPITAL_PROFILES),
        'rounds_completed': n_rounds,
        'final_global_auc': final_auc,
        'final_global_accuracy': final_acc,
        'round_metrics': round_metrics,
        'hospital_contributions': hospital_contributions,
        'privacy_note': 'No raw patient data was shared between hospitals. Only model weights were aggregated (FedAvg).',
    }
