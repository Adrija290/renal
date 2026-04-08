import os
from flask import Flask, jsonify, redirect, render_template_string, request, url_for
from ckd_engine import CKDBackend

app = Flask(__name__)
backend = CKDBackend()

FEATURE_INPUTS = [
    {'name': 'age', 'label': 'Age', 'type': 'number', 'step': '1', 'min': '1', 'max': '120', 'value': '45'},
    {'name': 'bp', 'label': 'Blood Pressure (mm/Hg)', 'type': 'number', 'step': '1', 'min': '40', 'max': '250', 'value': '80'},
    {'name': 'sg', 'label': 'Specific Gravity', 'type': 'select', 'options': ['1.005', '1.010', '1.015', '1.020', '1.025']},
    {'name': 'al', 'label': 'Albumin', 'type': 'select', 'options': ['0', '1', '2', '3', '4', '5']},
    {'name': 'su', 'label': 'Sugar', 'type': 'select', 'options': ['0', '1', '2', '3', '4', '5']},
    {'name': 'rbc', 'label': 'Red Blood Cells', 'type': 'select', 'options': ['normal', 'abnormal']},
    {'name': 'pc', 'label': 'Pus Cell', 'type': 'select', 'options': ['normal', 'abnormal']},
    {'name': 'pcc', 'label': 'Pus Cell Clumps', 'type': 'select', 'options': ['present', 'notpresent']},
    {'name': 'ba', 'label': 'Bacteria', 'type': 'select', 'options': ['present', 'notpresent']},
    {'name': 'bgr', 'label': 'Blood Glucose Random', 'type': 'number', 'step': '0.1', 'min': '0', 'max': '500', 'value': '120'},
    {'name': 'bu', 'label': 'Blood Urea', 'type': 'number', 'step': '0.1', 'min': '0', 'max': '200', 'value': '40'},
    {'name': 'sc', 'label': 'Serum Creatinine', 'type': 'number', 'step': '0.01', 'min': '0.1', 'max': '20', 'value': '1.2'},
    {'name': 'sod', 'label': 'Sodium', 'type': 'number', 'step': '0.1', 'min': '100', 'max': '180', 'value': '135'},
    {'name': 'pot', 'label': 'Potassium', 'type': 'number', 'step': '0.1', 'min': '2', 'max': '10', 'value': '4.5'},
    {'name': 'hemo', 'label': 'Hemoglobin', 'type': 'number', 'step': '0.1', 'min': '5', 'max': '20', 'value': '12'},
    {'name': 'pcv', 'label': 'Packed Cell Volume', 'type': 'number', 'step': '1', 'min': '10', 'max': '55', 'value': '40'},
    {'name': 'wc', 'label': 'White Blood Cell Count', 'type': 'number', 'step': '1', 'min': '2000', 'max': '30000', 'value': '8500'},
    {'name': 'rc', 'label': 'Red Blood Cell Count', 'type': 'number', 'step': '0.01', 'min': '1.0', 'max': '10.0', 'value': '4.5'},
    {'name': 'htn', 'label': 'Hypertension', 'type': 'select', 'options': ['yes', 'no']},
    {'name': 'dm', 'label': 'Diabetes Mellitus', 'type': 'select', 'options': ['yes', 'no']},
    {'name': 'cad', 'label': 'Coronary Artery Disease', 'type': 'select', 'options': ['yes', 'no']},
    {'name': 'appet', 'label': 'Appetite', 'type': 'select', 'options': ['good', 'poor']},
    {'name': 'pe', 'label': 'Pedal Edema', 'type': 'select', 'options': ['yes', 'no']},
    {'name': 'ane', 'label': 'Anemia', 'type': 'select', 'options': ['yes', 'no']},
]

TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CKD Prediction</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <style>
    body {
      font-family: 'Poppins', sans-serif;
      background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
      background-size: 400% 400%;
      animation: gradientBG 15s ease infinite;
      min-height: 100vh;
    }
    @keyframes gradientBG {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }
    .navbar-brand { font-weight: bold; }
    .logo-icon { color: #fff; margin-right: 10px; }
    .card { border: none; box-shadow: 0 10px 30px rgba(0,0,0,0.1); border-radius: 15px; backdrop-filter: blur(10px); background: rgba(255,255,255,0.9); }
    .fade-in { animation: fadeIn 1s ease-in; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .circular-progress {
      width: 150px; height: 150px; margin: 20px auto; position: relative;
    }
    .circular-progress svg { width: 100%; height: 100%; transform: rotate(-90deg); }
    .circular-progress circle { fill: none; stroke-width: 8; stroke-linecap: round; }
    .circular-progress .bg { stroke: #e9ecef; }
    .circular-progress .fg { stroke: #28a745; stroke-dasharray: 283; stroke-dashoffset: 283; transition: stroke-dashoffset 2s ease; }
    .circular-progress.high .fg { stroke: #dc3545; }
    .circular-progress .percentage { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 24px; font-weight: bold; }
    .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; padding: 15px; margin-bottom: 10px; text-align: center; }
    .stat-card:nth-child(2n) { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .stat-card:nth-child(3n) { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .stat-card:nth-child(4n) { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .footer { background: rgba(0,0,0,0.8); color: white; padding: 20px 0; margin-top: 40px; text-align: center; }
    .footer p { margin: 0; font-size: 14px; }
  </style>
</head>
<body class="bg-light">
  <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
    <div class="container">
      <i class="fas fa-kidneys logo-icon"></i>
      <a class="navbar-brand" href="#">CKD Prediction Tool</a>
    </div>
  </nav>

  <div class="container mt-4">
    <div class="row justify-content-center">
      <div class="col-lg-10">
        <div class="card mb-4 fade-in">
          <div class="card-body text-center">
            <i class="fas fa-kidneys fa-3x text-primary mb-3"></i>
            <h1 class="card-title">Chronic Kidney Disease Prediction</h1>
            <p class="text-muted">Enter patient features and submit to assess CKD risk using AI-powered analysis.</p>
          </div>
        </div>

        <div class="card mb-4 fade-in">
          <div class="card-body">
            <form method="post" action="{{ url_for('predict') }}">
              <div class="row g-3">
                {% for field in inputs %}
                  <div class="col-md-6 col-lg-4">
                    <label for="{{ field.name }}" class="form-label fw-bold">{{ field.label }}</label>
                    {% if field.type == 'select' %}
                      <select class="form-select" id="{{ field.name }}" name="{{ field.name }}" required>
                        {% for option in field.options %}
                          <option value="{{ option }}" {% if result and result.inputs[field.name] == option %}selected{% endif %}>{{ option }}</option>
                        {% endfor %}
                      </select>
                    {% else %}
                      <input class="form-control" id="{{ field.name }}" name="{{ field.name }}" type="{{ field.type }}" step="{{ field.step }}" min="{{ field.min }}" max="{{ field.max }}" value="{{ result.inputs[field.name] if result else field.value }}" required>
                    {% endif %}
                  </div>
                {% endfor %}
              </div>
              <div class="d-grid mt-4">
                <button class="btn btn-primary btn-lg" type="submit">
                  <i class="fas fa-chart-line me-2"></i>Predict CKD Risk
                </button>
              </div>
            </form>
          </div>
        </div>

        {% if result %}
          <div class="card result-card mb-4 fade-in {{ 'border-success' if result.prediction == 0 else 'border-danger' }}">
            <div class="card-body text-center">
              <h2 class="card-title mb-4">{{ result.label }}</h2>
              <div class="circular-progress {{ 'high' if result.prediction == 1 else '' }}">
                <svg>
                  <circle class="bg" cx="75" cy="75" r="45"></circle>
                  <circle class="fg" cx="75" cy="75" r="45" style="stroke-dashoffset: {{ 283 - (283 * result.confidence / 100) }}"></circle>
                </svg>
                <div class="percentage">{{ result.confidence }}%</div>
              </div>
              <p class="mb-3"><strong>Risk Confidence:</strong> {{ result.confidence }}%</p>
              <div class="alert {{ 'alert-success' if result.prediction == 0 else 'alert-danger' }} fade-in" role="alert">
                <i class="fas {{ 'fa-check-circle' if result.prediction == 0 else 'fa-exclamation-triangle' }} me-2"></i>
                {{ result.recommendation }}
              </div>
            </div>
          </div>

          <div class="card fade-in">
            <div class="card-body">
              <h3 class="card-title text-center mb-4"><i class="fas fa-clipboard-list me-2"></i>Patient Input Summary</h3>
              <div class="row g-3">
                {% for field in inputs %}
                  <div class="col-md-6 col-lg-4">
                    <div class="stat-card">
                      <h6 class="mb-1">{{ field.label }}</h6>
                      <h4 class="mb-0">{{ result.inputs[field.name] }}</h4>
                    </div>
                  </div>
                {% endfor %}
              </div>
            </div>
          </div>
        {% endif %}
      </div>
    </div>
  </div>

  <footer class="footer">
    <div class="container">
      <p><i class="fas fa-stethoscope me-2"></i>This tool is for educational purposes only. Always consult with a qualified healthcare professional for medical advice and diagnosis.</p>
      <p class="mt-2"><small>© 2026 CKD Prediction Tool | Powered by AI & Machine Learning</small></p>
    </div>
  </footer>

  <script>
    // Add fade-in animation to cards on load
    document.addEventListener('DOMContentLoaded', function() {
      const cards = document.querySelectorAll('.fade-in');
      cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
      });
    });
  </script>
</body>
</html>
'''


def _make_result(prediction, probability, inputs):
    label = 'Chronic Kidney Disease Likely' if prediction == 1 else 'Low CKD Risk'
    confidence = round(float(max(probability)) * 100, 1)
    recommendation = (
        "High risk detected. Please consult a nephrologist immediately for further evaluation and treatment."
        if prediction == 1 else
        "Low risk detected. Continue maintaining a healthy lifestyle with regular check-ups."
    )
    return {
        'prediction': prediction,
        'label': label,
        'confidence': confidence,
        'recommendation': recommendation,
        'raw_probability': {
            'notckd': round(float(probability[0]) * 100, 1),
            'ckd': round(float(probability[1]) * 100, 1),
        },
        'inputs': inputs,
    }


def _extract_form_values(form_data):
    values = {}
    for field in backend.feature_names:
        values[field] = form_data.get(field, '')
    return values


@app.route('/', methods=['GET'])
def home():
    return render_template_string(TEMPLATE, inputs=FEATURE_INPUTS, result=None)


@app.route('/predict', methods=['POST'])
def predict():
    patient_data = _extract_form_values(request.form)
    prediction, probability = backend.predict_patient(patient_data)
    result = _make_result(prediction, probability, patient_data)
    return render_template_string(TEMPLATE, inputs=FEATURE_INPUTS, result=result)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({'error': 'Invalid JSON payload'}), 400

    missing_fields = [name for name in backend.feature_names if name not in payload]
    if missing_fields:
        return jsonify({'error': 'Missing fields', 'missing': missing_fields}), 400

    prediction, probability = backend.predict_patient(payload)
    return jsonify({
        'prediction': 'ckd' if prediction == 1 else 'notckd',
        'probabilities': {
            'notckd': float(probability[0]),
            'ckd': float(probability[1]),
        },
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
