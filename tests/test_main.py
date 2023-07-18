from fastapi.testclient import TestClient
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))
from app import main

client = TestClient(main.app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "4Banks API!"}

def test_openapi_customization():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    json = response.json()
    assert json["info"]["title"] == "4banks API"
    assert json["info"]["version"] == "1.0.0"
    assert json["info"]["x-logo"]["url"] == "/flasgger_static/swagger-ui/logo_small.png"