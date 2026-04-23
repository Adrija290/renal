import datetime
from database.db import db


class Patient(db.Model):
    __tablename__ = 'patients'

    id = db.Column(db.String(20), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    sex = db.Column(db.String(1))
    race = db.Column(db.String(50))
    diabetes = db.Column(db.Boolean, default=False)
    hypertension = db.Column(db.Boolean, default=False)
    cad = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    lab_results = db.relationship('LabResult', backref='patient', lazy=True,
                                   order_by='LabResult.date')
    predictions = db.relationship('Prediction', backref='patient', lazy=True)
    symptoms = db.relationship('SymptomEntry', backref='patient', lazy=True)
    alerts = db.relationship('Alert', backref='patient', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'sex': self.sex,
            'race': self.race,
            'diabetes': self.diabetes,
            'hypertension': self.hypertension,
            'cad': self.cad,
        }


class LabResult(db.Model):
    __tablename__ = 'lab_results'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.id'), nullable=False)
    date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    egfr = db.Column(db.Float)
    creatinine = db.Column(db.Float)
    albumin = db.Column(db.Float)
    hemoglobin = db.Column(db.Float)
    potassium = db.Column(db.Float)
    sodium = db.Column(db.Float)
    blood_pressure_systolic = db.Column(db.Integer)
    blood_pressure_diastolic = db.Column(db.Integer)
    blood_glucose = db.Column(db.Float)
    uacr = db.Column(db.Float)

    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'date': self.date.isoformat() if self.date else None,
            'egfr': self.egfr,
            'creatinine': self.creatinine,
            'albumin': self.albumin,
            'hemoglobin': self.hemoglobin,
            'potassium': self.potassium,
            'sodium': self.sodium,
            'bp_systolic': self.blood_pressure_systolic,
            'bp_diastolic': self.blood_pressure_diastolic,
            'blood_glucose': self.blood_glucose,
            'uacr': self.uacr,
        }


class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.id'), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    prediction = db.Column(db.Integer)
    ckd_probability = db.Column(db.Float)
    confidence = db.Column(db.Float)
    model_version = db.Column(db.String(20), default='v1.0')
    input_features = db.Column(db.Text)
    shap_values = db.Column(db.Text)
    top_features = db.Column(db.Text)


class Alert(db.Model):
    __tablename__ = 'alerts'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    alert_type = db.Column(db.String(50))
    severity = db.Column(db.String(20))
    message = db.Column(db.Text)
    resolved = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'resolved': self.resolved,
        }


class SymptomEntry(db.Model):
    __tablename__ = 'symptom_entries'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    fatigue_score = db.Column(db.Integer)
    swelling_score = db.Column(db.Integer)
    nausea_score = db.Column(db.Integer)
    shortness_of_breath = db.Column(db.Boolean, default=False)
    confusion = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text)

    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'fatigue_score': self.fatigue_score,
            'swelling_score': self.swelling_score,
            'nausea_score': self.nausea_score,
            'shortness_of_breath': self.shortness_of_breath,
            'confusion': self.confusion,
            'notes': self.notes,
        }


class WearableReading(db.Model):
    __tablename__ = 'wearable_readings'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    device_type = db.Column(db.String(50))
    systolic_bp = db.Column(db.Integer)
    diastolic_bp = db.Column(db.Integer)
    heart_rate = db.Column(db.Integer)
    blood_glucose = db.Column(db.Float)
    weight_kg = db.Column(db.Float)
    steps = db.Column(db.Integer)


class AuditLog(db.Model):
    __tablename__ = 'audit_logs'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    event_type = db.Column(db.String(50))
    user_id = db.Column(db.String(50))
    patient_id = db.Column(db.String(20), nullable=True)
    action = db.Column(db.Text)
    ip_address = db.Column(db.String(50))
    outcome = db.Column(db.String(20))
    details = db.Column(db.Text)
