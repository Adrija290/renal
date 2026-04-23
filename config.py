import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'nephrocare-ai-dev-2026')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', f'sqlite:///{os.path.join(BASE_DIR, "ckd_system.db")}')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODEL_CACHE_DIR = os.path.join(BASE_DIR, 'models_cache')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    REFERENCE_DIR = os.path.join(BASE_DIR, 'data', 'reference')
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

    # GFR alert thresholds
    GFR_RAPID_DECLINE_THRESHOLD = 5.0   # mL/min/1.73m² per year
    ANOMALY_ZSCORE_THRESHOLD = 2.5
    DRIFT_KL_THRESHOLD = 0.15

    # Federated learning
    FL_NUM_HOSPITALS = 5
    FL_ROUNDS = 8

    # CKD stage boundaries (eGFR mL/min/1.73m²)
    CKD_STAGES = {
        'G1': (90, float('inf')),
        'G2': (60, 90),
        'G3a': (45, 60),
        'G3b': (30, 45),
        'G4': (15, 30),
        'G5': (0, 15),
    }

    # Dialysis preparation lead time (months before ESRD)
    DIALYSIS_PREP_LEAD_MONTHS = 6
