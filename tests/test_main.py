from fastapi.testclient import TestClient
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))
from app import main

client = TestClient(main.app)

def test_read_main():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'message': '4Banks API!'}

def test_openapi_customization():
    response = client.get('/openapi.json')
    assert response.status_code == 200
    json = response.json()
    assert json['info']['title'] == '4banks API'
    assert json['info']['version'] == '1.0.0'
    assert json['info']['x-logo']['url'] == '/flasgger_static/swagger-ui/logo_small.png'

def test_load_dataset():
    response = client.get('/datasets/test_id/test_file_0')
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 6
    expected_data = [
        {'V1': -1.359807134, 'V2': -0.072781173, 'V3': 2.536346738, 'V4': 1.378155224, 'Class': 0},
        {'V1': 1.191857111, 'V2': 0.266150712, 'V3': 0.166480113, 'V4': 0.448154078, 'Class': 0},
        {'V1': -1.358354062, 'V2': -1.340163075, 'V3': 1.773209343, 'V4': 0.379779593, 'Class': 0},
        {'V1': -0.966271712, 'V2': -0.185226008, 'V3': 1.79299334, 'V4': -0.863291275, 'Class': 0},
        {'V1': -1.158233093, 'V2': 0.877736755, 'V3': 1.548717847, 'V4': 0.403033934, 'Class': 1},
        {'V1': -0.425965884, 'V2': 0.960523045, 'V3': 1.141109342, 'V4': -0.16825208, 'Class': 1}
    ]
    assert data == expected_data

def test_balance_dataset():
    response = client.get('/datasets/test_id/test_file_0/balance?method=random_under_sampling')
    assert response.status_code == 200
    data = response.json()
    assert 'message' in data
    assert 'Resultado salvo com sucesso no seguinte local: gs://<BUCKET_NAME>/test_id/test_file_0_random_under_sampling.csv' == data['message']
