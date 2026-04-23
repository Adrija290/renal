"""
SHAP-based explainability for the CKD ensemble.
Returns feature importance and individual prediction explanations.
"""
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

FEATURE_LABELS = {
    'age': 'Age',
    'bp': 'Blood Pressure',
    'sg': 'Specific Gravity',
    'al': 'Albumin',
    'su': 'Sugar',
    'bgr': 'Blood Glucose',
    'bu': 'Blood Urea',
    'sc': 'Serum Creatinine',
    'sod': 'Sodium',
    'pot': 'Potassium',
    'hemo': 'Hemoglobin',
    'pcv': 'Packed Cell Volume',
    'wc': 'WBC Count',
    'rc': 'RBC Count',
    'rbc': 'RBC Morphology',
    'pc': 'Pus Cells',
    'pcc': 'Pus Cell Clumps',
    'ba': 'Bacteria',
    'htn': 'Hypertension',
    'dm': 'Diabetes',
    'cad': 'Coronary Artery Disease',
    'appet': 'Appetite',
    'pe': 'Pedal Edema',
    'ane': 'Anemia',
}


class ShapExplainer:
    """Compute SHAP values for individual predictions."""

    def __init__(self, ensemble_model):
        self.ensemble = ensemble_model
        self._explainer = None

    def _get_explainer(self, X_background: np.ndarray):
        if not HAS_SHAP:
            return None
        if self._explainer is not None:
            return self._explainer

        rf_model = None
        for name, estimator in self.ensemble.ensemble.estimators:
            if name == 'rf':
                rf_model = estimator
                break

        if rf_model is not None:
            self._explainer = shap.TreeExplainer(rf_model)
        else:
            self._explainer = shap.KernelExplainer(
                self.ensemble.ensemble.predict_proba,
                shap.sample(X_background, 50)
            )
        return self._explainer

    def explain(self, patient_data: dict, top_n: int = 8) -> dict:
        """
        Returns top_n features driving the prediction with their SHAP values.
        Falls back to RF feature importance if SHAP unavailable.
        """
        if not self.ensemble.is_trained:
            self.ensemble.load()

        df = pd.DataFrame([patient_data])
        df = self.ensemble.preprocess_features(df)
        from models.ensemble import ALL_FEATURES
        X = df[ALL_FEATURES]
        X_t = self.ensemble.preprocessor.transform(X)

        if HAS_SHAP:
            try:
                explainer = self._get_explainer(X_t)
                shap_values = explainer.shap_values(X_t)

                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                else:
                    sv = shap_values[0]

                feature_names = self._get_transformed_feature_names()
                feature_shap = list(zip(feature_names, sv.tolist()))
                feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
                top_features = feature_shap[:top_n]

                return {
                    'method': 'SHAP',
                    'top_features': [
                        {
                            'feature': f,
                            'label': FEATURE_LABELS.get(f.split('_')[0], f),
                            'shap_value': round(v, 4),
                            'direction': 'increases_risk' if v > 0 else 'decreases_risk',
                        }
                        for f, v in top_features
                    ],
                    'base_value': float(explainer.expected_value[1]
                                        if isinstance(explainer.expected_value, list)
                                        else explainer.expected_value),
                }
            except Exception:
                pass

        return self._feature_importance_fallback(top_n)

    def _get_transformed_feature_names(self) -> list:
        from models.ensemble import NUMERIC_FEATURES, CATEGORICAL_FEATURES
        numeric_names = NUMERIC_FEATURES
        cat_encoder = self.ensemble.preprocessor.named_transformers_['cat']['onehot']
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        return numeric_names + cat_names

    def _feature_importance_fallback(self, top_n: int) -> dict:
        from models.ensemble import ALL_FEATURES
        importances = []
        for name, estimator in self.ensemble.ensemble.estimators:
            if hasattr(estimator, 'feature_importances_'):
                importances.append(estimator.feature_importances_)

        if not importances:
            return {'method': 'unavailable', 'top_features': []}

        avg_imp = np.mean(importances, axis=0)
        feature_names = self._get_transformed_feature_names()
        ranked = sorted(zip(feature_names, avg_imp.tolist()), key=lambda x: x[1], reverse=True)

        return {
            'method': 'FeatureImportance',
            'top_features': [
                {
                    'feature': f,
                    'label': FEATURE_LABELS.get(f.split('_')[0], f),
                    'importance': round(v, 4),
                    'direction': 'contributes_to_risk',
                }
                for f, v in ranked[:top_n]
            ],
        }
