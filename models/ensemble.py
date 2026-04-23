"""
Multi-model ensemble: RandomForest + XGBoost + LightGBM + MLP with soft voting.
Gracefully degrades to available models if optional deps are missing.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from config import Config

NUMERIC_FEATURES = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
                     'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
CATEGORICAL_FEATURES = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_PATH = os.path.join(Config.MODEL_CACHE_DIR, 'ensemble_model.joblib')
META_PATH = os.path.join(Config.MODEL_CACHE_DIR, 'ensemble_meta.json')


def _build_preprocessor():
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', numeric_pipe, NUMERIC_FEATURES),
        ('cat', categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder='drop', sparse_threshold=0)


def _build_estimators():
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=12,
                                       min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                               early_stopping=True, random_state=42)),
    ]
    if HAS_XGB:
        estimators.append(('xgb', XGBClassifier(n_estimators=200, max_depth=6,
                                                  learning_rate=0.05, use_label_encoder=False,
                                                  eval_metric='logloss', random_state=42,
                                                  verbosity=0)))
    if HAS_LGBM:
        estimators.append(('lgbm', LGBMClassifier(n_estimators=200, max_depth=6,
                                                    learning_rate=0.05, random_state=42,
                                                    verbose=-1)))
    return estimators


class CKDEnsemble:
    def __init__(self):
        self.preprocessor = _build_preprocessor()
        self.ensemble = None
        self.is_trained = False
        self.metrics = {}
        self.feature_names = ALL_FEATURES
        self.numeric_features = NUMERIC_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)

    def _normalize_value(self, value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip().lower()
        mapping = {
            'yes': 1, 'no': 0,
            'present': 1, 'notpresent': 0,
            'normal': 0, 'abnormal': 1,
            'good': 1, 'poor': 0,
        }
        return mapping.get(text, text)

    def preprocess_features(self, df):
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        df.replace('?', np.nan, inplace=True)
        df = df.map(self._normalize_value)
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(object)
        return df

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = self.preprocess_features(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        X_tr_t = self.preprocessor.fit_transform(X_tr)
        X_te_t = self.preprocessor.transform(X_te)

        estimators = _build_estimators()
        self.ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        self.ensemble.fit(X_tr_t, y_tr)
        self.is_trained = True

        y_pred = self.ensemble.predict(X_te_t)
        y_prob = self.ensemble.predict_proba(X_te_t)[:, 1]
        report = classification_report(y_te, y_pred, output_dict=True)
        self.metrics = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score'],
            'auc_roc': round(roc_auc_score(y_te, y_prob), 4),
            'models_used': [e[0] for e in estimators],
        }
        self.save()
        return self.metrics

    def predict(self, patient_data: dict):
        if not self.is_trained:
            self.load()
        df = pd.DataFrame([patient_data])
        df = self.preprocess_features(df)
        X = df[ALL_FEATURES]
        X_t = self.preprocessor.transform(X)
        prediction = int(self.ensemble.predict(X_t)[0])
        probability = self.ensemble.predict_proba(X_t)[0]
        return prediction, probability

    def save(self):
        joblib.dump({'preprocessor': self.preprocessor, 'ensemble': self.ensemble}, MODEL_PATH)
        with open(META_PATH, 'w') as f:
            json.dump(self.metrics, f)

    def load(self):
        if os.path.exists(MODEL_PATH):
            data = joblib.load(MODEL_PATH)
            self.preprocessor = data['preprocessor']
            self.ensemble = data['ensemble']
            self.is_trained = True
            if os.path.exists(META_PATH):
                with open(META_PATH) as f:
                    self.metrics = json.load(f)
        else:
            from ckd_engine import CKDBackend
            backend = CKDBackend()
            df = backend.load_data()
            X = df[ALL_FEATURES]
            y = df['target'].astype(int)
            self.fit(X, y)
