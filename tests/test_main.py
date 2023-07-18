from fastapi.testclient import TestClient
from fastapi import UploadFile
import pandas as pd
from io import StringIO
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "4Banks API!"}

def test_openapi_customization():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    json = response.json()
    assert json["info"]["title"] == "Your application name"
    assert json["info"]["version"] == "1.0.0"
    assert json["info"]["description"] == "This is a very custom OpenAPI schema"
    assert json["info"]["x-logo"]["url"] == "/flasgger_static/swagger-ui/logo_small.png"