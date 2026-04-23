"""
FHIR R4 connector for EHR integration (Epic, Cerner).
Parses FHIR bundles and extracts CKD-relevant observations.
Generates FHIR-compliant output bundles.
"""
import json
import datetime
import uuid

LOINC_MAP = {
    '33914-3': 'egfr',
    '2160-0': 'creatinine',
    '1751-7': 'albumin',
    '718-7': 'hemoglobin',
    '2823-3': 'potassium',
    '2951-2': 'sodium',
    '14959-1': 'uacr',
    '2339-0': 'blood_glucose',
    '55284-4': 'blood_pressure',
    '8480-6': 'bp_systolic',
    '8462-4': 'bp_diastolic',
    '29463-7': 'weight_kg',
    '8867-4': 'heart_rate',
}

EHR_SYSTEMS = {
    'epic': 'https://open.epic.com/Interface/FHIR',
    'cerner': 'https://fhir.cerner.com/r4',
    'custom': 'https://fhir.example.org/r4',
}


def parse_fhir_bundle(bundle_json: str | dict) -> dict:
    """
    Parse a FHIR R4 Bundle containing Patient and Observation resources.
    Returns structured patient and lab data.
    """
    if isinstance(bundle_json, str):
        bundle = json.loads(bundle_json)
    else:
        bundle = bundle_json

    if bundle.get('resourceType') != 'Bundle':
        return {'error': f"Expected Bundle, got {bundle.get('resourceType')}"}

    patient_data = {}
    observations = []
    conditions = []

    for entry in bundle.get('entry', []):
        resource = entry.get('resource', {})
        rt = resource.get('resourceType')

        if rt == 'Patient':
            patient_data = _parse_patient(resource)
        elif rt == 'Observation':
            obs = _parse_observation(resource)
            if obs:
                observations.append(obs)
        elif rt == 'Condition':
            cond = _parse_condition(resource)
            if cond:
                conditions.append(cond)

    # Map observations to lab fields
    lab_results = {}
    for obs in observations:
        field = LOINC_MAP.get(obs.get('loinc_code'), obs.get('loinc_code'))
        if field and obs.get('value') is not None:
            lab_results[field] = obs['value']

    return {
        'patient': patient_data,
        'lab_results': lab_results,
        'observations_raw': observations,
        'conditions': conditions,
        'parsed_at': datetime.datetime.utcnow().isoformat(),
    }


def _parse_patient(resource: dict) -> dict:
    name_list = resource.get('name', [{}])
    name_obj = name_list[0] if name_list else {}
    family = name_obj.get('family', '')
    given = ' '.join(name_obj.get('given', []))
    full_name = f"{given} {family}".strip()

    birth_date = resource.get('birthDate', '')
    age = None
    if birth_date:
        try:
            dob = datetime.datetime.strptime(birth_date, '%Y-%m-%d')
            age = int((datetime.datetime.now() - dob).days / 365.25)
        except ValueError:
            pass

    gender_map = {'male': 'M', 'female': 'F', 'other': 'O', 'unknown': 'U'}
    sex = gender_map.get(resource.get('gender', '').lower(), 'U')

    race = ''
    for ext in resource.get('extension', []):
        url = ext.get('url', '')
        if 'us-core-race' in url:
            for sub_ext in ext.get('extension', []):
                if sub_ext.get('url') == 'text':
                    race = sub_ext.get('valueString', '')

    return {
        'fhir_id': resource.get('id', ''),
        'name': full_name,
        'age': age,
        'sex': sex,
        'race': race,
        'birth_date': birth_date,
    }


def _parse_observation(resource: dict) -> dict | None:
    code_obj = resource.get('code', {})
    codings = code_obj.get('coding', [])
    loinc_code = None
    for coding in codings:
        if coding.get('system', '').lower().endswith('loinc.org'):
            loinc_code = coding.get('code')
            break

    if loinc_code not in LOINC_MAP and loinc_code != '55284-4':
        return None

    value = None
    unit = ''
    if 'valueQuantity' in resource:
        value = resource['valueQuantity'].get('value')
        unit = resource['valueQuantity'].get('unit', '')
    elif 'valueCodeableConcept' in resource:
        value = resource['valueCodeableConcept'].get('text')
    elif 'component' in resource:
        components = resource['component']
        result = {}
        for comp in components:
            comp_code = comp.get('code', {}).get('coding', [{}])[0].get('code', '')
            comp_val = comp.get('valueQuantity', {}).get('value')
            if comp_code == '8480-6':
                result['systolic'] = comp_val
            elif comp_code == '8462-4':
                result['diastolic'] = comp_val
        return {'loinc_code': loinc_code, 'value': result, 'unit': 'mmHg',
                'date': resource.get('effectiveDateTime', '')}

    date = resource.get('effectiveDateTime') or resource.get('issued', '')

    return {
        'loinc_code': loinc_code,
        'value': value,
        'unit': unit,
        'date': date,
        'status': resource.get('status', ''),
    }


def _parse_condition(resource: dict) -> dict | None:
    code_obj = resource.get('code', {})
    codings = code_obj.get('coding', [])
    icd_code = None
    display = code_obj.get('text', '')
    for coding in codings:
        if 'icd' in coding.get('system', '').lower():
            icd_code = coding.get('code')
        display = display or coding.get('display', '')

    if not icd_code and not display:
        return None

    clinical_status = (resource.get('clinicalStatus', {})
                       .get('coding', [{}])[0]
                       .get('code', 'unknown'))

    return {
        'icd_code': icd_code,
        'display': display,
        'clinical_status': clinical_status,
    }


def generate_fhir_observation(patient_fhir_id: str, loinc_code: str,
                               value: float, unit: str,
                               date: str = None) -> dict:
    """Generate a FHIR R4 Observation resource for a lab result."""
    date = date or datetime.datetime.utcnow().isoformat()
    loinc_displays = {
        '33914-3': 'eGFR (CKD-EPI)',
        '2160-0': 'Serum Creatinine',
        '1751-7': 'Albumin, Serum',
        '718-7': 'Hemoglobin [Mass/volume] in Blood',
        '2823-3': 'Potassium [Moles/volume] in Serum',
        '2951-2': 'Sodium [Moles/volume] in Serum',
        '14959-1': 'Microalbumin/Creatinine [Ratio] in Urine',
    }
    return {
        'resourceType': 'Observation',
        'id': str(uuid.uuid4()),
        'status': 'final',
        'code': {
            'coding': [{
                'system': 'http://loinc.org',
                'code': loinc_code,
                'display': loinc_displays.get(loinc_code, loinc_code),
            }],
            'text': loinc_displays.get(loinc_code, loinc_code),
        },
        'subject': {'reference': f'Patient/{patient_fhir_id}'},
        'effectiveDateTime': date,
        'valueQuantity': {'value': value, 'unit': unit,
                           'system': 'http://unitsofmeasure.org'},
    }


def fhir_demo_bundle() -> dict:
    """Returns a sample FHIR Bundle for testing the connector."""
    return {
        'resourceType': 'Bundle',
        'type': 'collection',
        'entry': [
            {
                'resource': {
                    'resourceType': 'Patient',
                    'id': 'patient-demo-001',
                    'name': [{'family': 'Smith', 'given': ['John', 'A']}],
                    'gender': 'male',
                    'birthDate': '1962-04-15',
                    'extension': [{
                        'url': 'http://hl7.org/fhir/us/core/StructureDefinition/us-core-race',
                        'extension': [{'url': 'text', 'valueString': 'White'}]
                    }]
                }
            },
            {'resource': {'resourceType': 'Observation', 'status': 'final',
                           'code': {'coding': [{'system': 'http://loinc.org', 'code': '33914-3', 'display': 'eGFR'}]},
                           'valueQuantity': {'value': 38.5, 'unit': 'mL/min/1.73m2'},
                           'effectiveDateTime': '2026-01-15T10:30:00Z'}},
            {'resource': {'resourceType': 'Observation', 'status': 'final',
                           'code': {'coding': [{'system': 'http://loinc.org', 'code': '2160-0', 'display': 'Creatinine'}]},
                           'valueQuantity': {'value': 2.1, 'unit': 'mg/dL'},
                           'effectiveDateTime': '2026-01-15T10:30:00Z'}},
            {'resource': {'resourceType': 'Observation', 'status': 'final',
                           'code': {'coding': [{'system': 'http://loinc.org', 'code': '718-7', 'display': 'Hemoglobin'}]},
                           'valueQuantity': {'value': 10.2, 'unit': 'g/dL'},
                           'effectiveDateTime': '2026-01-15T10:30:00Z'}},
            {'resource': {'resourceType': 'Condition',
                           'code': {'coding': [{'system': 'http://hl7.org/fhir/sid/icd-10', 'code': 'N18.3', 'display': 'Chronic kidney disease, stage 3'}],
                                    'text': 'CKD Stage 3'},
                           'clinicalStatus': {'coding': [{'code': 'active'}]}}},
        ]
    }
