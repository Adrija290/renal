import json
import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from database.db import Base, db_session
from database.models import PredictionLog

class AuditTrail(Base):
    __tablename__ = 'audit_trail'
    
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey('prediction_log.id'))
    action = Column(String(100))
    user_id = Column(String(100))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    details = Column(Text)
    ip_address = Column(String(45))

def log_prediction_audit(prediction_id, action, user_id, details, ip_address):
    """Log audit trail for predictions"""
    audit = AuditTrail(
        prediction_id=prediction_id,
        action=action,
        user_id=user_id,
        details=json.dumps(details),
        ip_address=ip_address
    )
    db_session.add(audit)
    db_session.commit()
    return audit.id

def get_audit_trail(period_days=30):
    """Retrieve recent audit trail"""
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=period_days)
    return db_session.query(AuditTrail).filter(AuditTrail.timestamp > cutoff).all()

def export_audit_trail(filename='audit_trail.json'):
    """Export audit trail to JSON for regulatory compliance"""
    audits = get_audit_trail()
    data = [{
        'id': a.id,
        'prediction_id': a.prediction_id,
        'action': a.action,
        'user_id': a.user_id,
        'timestamp': a.timestamp.isoformat(),
        'details': json.loads(a.details),
        'ip_address': a.ip_address
    } for a in audits]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    return filename

