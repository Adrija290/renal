import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CKDBackend:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=200, random_state=random_state)
        self.feature_names = [
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc',
            'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]
        self.numeric_features = [
            'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot',
            'hemo', 'pcv', 'wc', 'rc'
        ]
        self.categorical_features = [
            'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]
        self.pipeline = None
        self.is_trained = False
        self.report = None
        self._build_transformer()

    def _build_transformer(self):
        numeric_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features),
            ],
            remainder='drop',
            sparse_threshold=0,
        )

    def load_data(self):
        local_path = os.path.join(os.path.dirname(__file__), 'data', 'chronic_kidney_disease.csv')
        if os.path.exists(local_path):
            raw_df = pd.read_csv(local_path)
        else:
            raw_df = pd.DataFrame([
                {
                    'age': 48, 'bp': 80, 'sg': 1.020, 'al': 1, 'su': 0,
                    'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
                    'bgr': 121, 'bu': 36, 'sc': 1.2, 'sod': 135, 'pot': 4.5,
                    'hemo': 15.4, 'pcv': 44, 'wc': 7800, 'rc': 5.2,
                    'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good',
                    'pe': 'no', 'ane': 'no', 'class': 'notckd'
                },
                {
                    'age': 62, 'bp': 90, 'sg': 1.010, 'al': 4, 'su': 1,
                    'rbc': 'abnormal', 'pc': 'abnormal', 'pcc': 'present', 'ba': 'notpresent',
                    'bgr': 120, 'bu': 57, 'sc': 1.7, 'sod': 138, 'pot': 4.2,
                    'hemo': 9.9, 'pcv': 30, 'wc': 6700, 'rc': 3.9,
                    'htn': 'yes', 'dm': 'yes', 'cad': 'no', 'appet': 'poor',
                    'pe': 'yes', 'ane': 'yes', 'class': 'ckd'
                },
                {
                    'age': 37, 'bp': 70, 'sg': 1.015, 'al': 0, 'su': 0,
                    'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
                    'bgr': 85, 'bu': 23, 'sc': 0.9, 'sod': 142, 'pot': 4.8,
                    'hemo': 13.5, 'pcv': 40, 'wc': 7000, 'rc': 5.0,
                    'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good',
                    'pe': 'no', 'ane': 'no', 'class': 'notckd'
                },
                {
                    'age': 58, 'bp': 100, 'sg': 1.015, 'al': 2, 'su': 1,
                    'rbc': 'abnormal', 'pc': 'abnormal', 'pcc': 'present', 'ba': 'present',
                    'bgr': 208, 'bu': 72, 'sc': 3.2, 'sod': 128, 'pot': 4.9,
                    'hemo': 11.3, 'pcv': 33, 'wc': 7200, 'rc': 4.0,
                    'htn': 'yes', 'dm': 'yes', 'cad': 'yes', 'appet': 'poor',
                    'pe': 'yes', 'ane': 'yes', 'class': 'ckd'
                },
                {
                    'age': 45, 'bp': 72, 'sg': 1.020, 'al': 1, 'su': 0,
                    'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
                    'bgr': 100, 'bu': 30, 'sc': 1.1, 'sod': 136, 'pot': 4.1,
                    'hemo': 14.2, 'pcv': 42, 'wc': 7600, 'rc': 5.0,
                    'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good',
                    'pe': 'no', 'ane': 'no', 'class': 'notckd'
                },
                {
                    'age': 55, 'bp': 85, 'sg': 1.018, 'al': 3, 'su': 2,
                    'rbc': 'abnormal', 'pc': 'abnormal', 'pcc': 'present', 'ba': 'notpresent',
                    'bgr': 150, 'bu': 45, 'sc': 2.1, 'sod': 140, 'pot': 4.0,
                    'hemo': 10.5, 'pcv': 35, 'wc': 8000, 'rc': 4.2,
                    'htn': 'yes', 'dm': 'yes', 'cad': 'no', 'appet': 'poor',
                    'pe': 'no', 'ane': 'no', 'class': 'ckd'
                },
                {
                    'age': 42, 'bp': 75, 'sg': 1.025, 'al': 0, 'su': 0,
                    'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
                    'bgr': 90, 'bu': 25, 'sc': 0.8, 'sod': 145, 'pot': 4.3,
                    'hemo': 14.8, 'pcv': 45, 'wc': 7500, 'rc': 5.1,
                    'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good',
                    'pe': 'no', 'ane': 'no', 'class': 'notckd'
                },
                {
                    'age': 65, 'bp': 95, 'sg': 1.012, 'al': 4, 'su': 3,
                    'rbc': 'abnormal', 'pc': 'abnormal', 'pcc': 'present', 'ba': 'present',
                    'bgr': 220, 'bu': 80, 'sc': 4.0, 'sod': 130, 'pot': 5.2,
                    'hemo': 8.5, 'pcv': 28, 'wc': 9000, 'rc': 3.5,
                    'htn': 'yes', 'dm': 'yes', 'cad': 'yes', 'appet': 'poor',
                    'pe': 'yes', 'ane': 'yes', 'class': 'ckd'
                },
                {
                    'age': 50, 'bp': 78, 'sg': 1.022, 'al': 1, 'su': 0,
                    'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
                    'bgr': 110, 'bu': 32, 'sc': 1.0, 'sod': 137, 'pot': 4.4,
                    'hemo': 15.0, 'pcv': 43, 'wc': 7700, 'rc': 5.3,
                    'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good',
                    'pe': 'no', 'ane': 'no', 'class': 'notckd'
                },
                {
                    'age': 60, 'bp': 88, 'sg': 1.014, 'al': 2, 'su': 1,
                    'rbc': 'abnormal', 'pc': 'abnormal', 'pcc': 'present', 'ba': 'notpresent',
                    'bgr': 180, 'bu': 60, 'sc': 2.5, 'sod': 132, 'pot': 4.7,
                    'hemo': 11.0, 'pcv': 32, 'wc': 8200, 'rc': 4.1,
                    'htn': 'yes', 'dm': 'yes', 'cad': 'no', 'appet': 'poor',
                    'pe': 'yes', 'ane': 'no', 'class': 'ckd'
                },
            ])

        return self._clean_dataframe(raw_df)

    def _clean_dataframe(self, df):
        df = df.copy()
        df.columns = [col.strip().lower() for col in df.columns]
        df.replace('?', np.nan, inplace=True)

        if 'class' in df.columns:
            df.rename(columns={'class': 'target'}, inplace=True)

        expected_cols = set(self.feature_names + ['target'])
        if not expected_cols.issubset(set(df.columns)):
            missing = expected_cols - set(df.columns)
            raise ValueError(f'Missing expected CKD columns: {sorted(missing)}')

        df = df[self.feature_names + ['target']]

        df[self.feature_names] = self._preprocess_features(df[self.feature_names])[self.feature_names]

        df['target'] = df['target'].astype(str).str.strip().str.lower().map({'ckd': 1, 'notckd': 0, '1': 1, '0': 0})

        if df['target'].isna().any():
            raise ValueError('Target values must be ckd/notckd or 1/0')

        return df

    @staticmethod
    def _normalize_value(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip().lower()
        boolean_map = {
            'yes': 1,
            'no': 0,
            'present': 1,
            'notpresent': 0,
            'normal': 0,
            'abnormal': 1,
            'good': 1,
            'poor': 0,
        }
        if text in boolean_map:
            return boolean_map[text]
        return text

    def train(self):
        df = self.load_data()
        X = df[self.feature_names]
        y = df['target'].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        self.pipeline.fit(X_train)
        X_train_transformed = self.pipeline.transform(X_train)

        self.model.fit(X_train_transformed, y_train)
        self.is_trained = True

        X_test_transformed = self.pipeline.transform(X_test)
        y_pred = self.model.predict(X_test_transformed)
        self.report = classification_report(y_test, y_pred, output_dict=True)
        return self.report

    def _preprocess_features(self, df):
        df = df.copy()
        df.columns = [col.strip().lower() for col in df.columns]
        df.replace('?', np.nan, inplace=True)

        df = df.map(self._normalize_value)

        for col in self.numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(object)

        return df

    def predict_patient(self, patient_data):
        if not self.is_trained:
            self.train()

        if isinstance(patient_data, dict):
            row = {key: patient_data.get(key, np.nan) for key in self.feature_names}
        else:
            row = dict(zip(self.feature_names, patient_data))

        df = pd.DataFrame([row])
        df = self._preprocess_features(df)
        X = df[self.feature_names]
        X_transformed = self.pipeline.transform(X)

        probability = self.model.predict_proba(X_transformed)[0]
        prediction = int(self.model.predict(X_transformed)[0])
        return prediction, probability
