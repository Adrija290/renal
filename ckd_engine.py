import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
# from database.db import Base, engine, db_session
from database.models import *  # All models
from models.ensemble import EnsemblePredictor
from models.survival_model import SurvivalAnalyzer
from models.gfr_forecaster import GFRForecaster
from models.explainability import SHAPExplainer
from risk_monitoring.gfr_tracker import GFRTracker
from risk_monitoring.anomaly_detector import AnomalyDetector
from risk_monitoring.comorbidity_scorer import ComorbidityScorer
from clinical_support.treatment_engine import TreatmentEngine
from clinical_support.drug_checker import DrugChecker
from clinical_support.dialysis_planner import DialysisPlanner
from patient_portal.risk_dashboard import RiskDashboard
from patient_portal.diet_planner import DietPlanner
from patient_portal.symptom_tracker import SymptomTracker
from research.federated_learning import FederatedLearner
from research.cohort_analytics import CohortAnalyzer
from research.trial_matcher import TrialMatcher
from data_integration.fhir_connector import FHIRConnector
from governance.audit_trail import log_prediction_audit
from governance.fairness_engine import run_fairness_audit
from governance.drift_monitor import DriftMonitor

class CKDOrchestrator:
    def __init__(self):
        self.ensemble = EnsemblePredictor()
        self.survival = SurvivalAnalyzer()
        self.gfr_forecast = GFRForecaster()
        self.explainer = SHAPExplainer()
        self.gfr_tracker = GFRTracker()
        self.anomaly_detector = AnomalyDetector()
        self.comorbidity_scorer = ComorbidityScorer()
        self.treatment_engine = TreatmentEngine()
        self.drug_checker = DrugChecker()
        self.dialysis_planner = DialysisPlanner()
        self.risk_dashboard = RiskDashboard()
        self.diet_planner = DietPlanner()
        self.symptom_tracker = SymptomTracker()
        self.federated = FederatedLearner()
        self.cohort = CohortAnalyzer()
        self.trial_matcher = TrialMatcher()
        self.fhir = FHIRConnector()
        self.drift_monitor = DriftMonitor()
        self._initialize_demo_data()

    def _initialize_demo_data(self):
        pass  # DB init moved to Flask app context


    def predict_comprehensive(self, patient_data, user_id='demo_user'):
        # Core prediction
        risk_score = self.ensemble.predict(patient_data)['risk_probability']
        shap_values = self.explainer.explain(patient_data)
        
        # Survival & forecast
        survival_risk = self.survival.predict_survival(patient_data)
        gfr_trajectory = self.gfr_forecast.forecast_gfr(patient_data)
        
        # Risk monitoring
        anomaly_score = self.anomaly_detector.detect_anomalies(patient_data)
        comorbidity_risks = self.comorbidity_scorer.score_comorbidities(patient_data)
        gfr_trend = self.gfr_tracker.track_progression(patient_data)
        
        # Clinical support
        treatment_recs = self.treatment_engine.get_recommendations(patient_data)
        drug_alerts = self.drug_checker.check_drugs(patient_data)
        dialysis_window = self.dialysis_planner.predict_dialysis_timing(patient_data)
        
        # Patient features
        risk_profile = self.risk_dashboard.generate_dashboard(patient_data)
        diet_plan = self.diet_planner.generate_diet(patient_data)
        
        # Research
        trial_matches = self.trial_matcher.match_trials(patient_data)
        cohort_insights = self.cohort.analyze_cohort(patient_data)
        
        # Governance
        audit_id = log_prediction_audit(0, 'comprehensive_predict', user_id, {'risk_score': risk_score}, '127.0.0.1')
        fairness_report = run_fairness_audit()
        drift_status = self.drift_monitor.detect_concept_drift(pd.DataFrame([patient_data]))
        
        return {
            'core': {'risk_score': risk_score, 'shap_values': shap_values},
            'forecast': {'survival_risk': survival_risk, 'gfr_trajectory': gfr_trajectory},
            'risk_monitoring': {'anomaly_score': anomaly_score, 'comorbidities': comorbidity_risks, 'gfr_trend': gfr_trend},
            'clinical_support': {'treatments': treatment_recs, 'drug_alerts': drug_alerts, 'dialysis_window': dialysis_window},
            'patient': {'risk_profile': risk_profile, 'diet_plan': diet_plan},
            'research': {'trial_matches': trial_matches, 'cohort_insights': cohort_insights},
            'governance': {'audit_id': audit_id, 'fairness': fairness_report, 'drift': drift_status}
        }

    # Backward compatibility
    def predict_patient(self, patient_data):
        result = self.predict_comprehensive(patient_data)
        prediction = 1 if result['core']['risk_score'] > 0.5 else 0
        probability = [1 - result['core']['risk_score'], result['core']['risk_score']]
        return prediction, probability

orchestrator = CKDOrchestrator()

