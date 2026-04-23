from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from database.db import db, init_db
import os

app = Flask(__name__)
app.config.from_object('config.Config')
db.init_app(app)

from ckd_engine import orchestrator

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    patient_data = request.form.to_dict()
    result = orchestrator.predict_comprehensive(patient_data)
    return render_template('index.html', result=result)

@app.route('/clinical')
def clinical():
    return render_template('clinical.html')

@app.route('/patient')
def patient():
    return render_template('patient.html')

@app.route('/research')
def research():
    return render_template('research.html')

@app.route('/governance')
def governance():
    return render_template('governance.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    result = orchestrator.predict_comprehensive(data)
    return jsonify(result)

if __name__ == '__main__':
    init_db(app)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

