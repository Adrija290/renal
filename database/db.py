from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
        _seed_demo_data()


def _seed_demo_data():
    from database.models import Patient, LabResult
    if Patient.query.count() > 0:
        return

    import datetime
    import random
    random.seed(42)

    demo_patients = [
        Patient(id='P001', name='Alice Johnson', age=62, sex='F', race='White',
                diabetes=True, hypertension=True, cad=False,
                created_at=datetime.datetime.utcnow()),
        Patient(id='P002', name='Robert Chen', age=55, sex='M', race='Asian',
                diabetes=True, hypertension=True, cad=True,
                created_at=datetime.datetime.utcnow()),
        Patient(id='P003', name='Maria Garcia', age=48, sex='F', race='Hispanic',
                diabetes=False, hypertension=True, cad=False,
                created_at=datetime.datetime.utcnow()),
        Patient(id='P004', name='James Wilson', age=70, sex='M', race='Black',
                diabetes=True, hypertension=True, cad=True,
                created_at=datetime.datetime.utcnow()),
        Patient(id='P005', name='Sarah Kim', age=38, sex='F', race='Asian',
                diabetes=False, hypertension=False, cad=False,
                created_at=datetime.datetime.utcnow()),
    ]

    # GFR trajectories: declining over 24 months
    trajectories = {
        'P001': [45, 43, 41, 40, 38, 36, 35, 33],
        'P002': [32, 31, 30, 28, 27, 26, 25, 24],
        'P003': [68, 66, 65, 63, 62, 60, 59, 57],
        'P004': [22, 21, 20, 18, 17, 16, 15, 14],
        'P005': [88, 87, 88, 86, 85, 84, 83, 82],
    }

    for patient in demo_patients:
        db.session.add(patient)

    db.session.flush()

    for patient in demo_patients:
        gfrs = trajectories[patient.id]
        for i, gfr_val in enumerate(gfrs):
            months_ago = (len(gfrs) - 1 - i) * 3
            lab_date = datetime.datetime.utcnow() - datetime.timedelta(days=months_ago * 30)
            lab = LabResult(
                patient_id=patient.id,
                date=lab_date,
                egfr=gfr_val + random.uniform(-1, 1),
                creatinine=round(random.uniform(1.2, 4.5), 2),
                albumin=round(random.uniform(2.5, 4.5), 1),
                hemoglobin=round(random.uniform(8.5, 14.5), 1),
                potassium=round(random.uniform(3.8, 5.5), 1),
                sodium=round(random.uniform(132, 142), 1),
                blood_pressure_systolic=random.randint(120, 170),
                blood_pressure_diastolic=random.randint(70, 100),
                blood_glucose=random.randint(90, 250),
                uacr=round(random.uniform(30, 3000), 0),
            )
            db.session.add(lab)

    db.session.commit()
