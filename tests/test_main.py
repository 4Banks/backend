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

def test_upload_file():
    data = {
        "name": ["John", "Anna", "Peter", "Linda"],
        "age": [23, 23, 34, 45],
    }
    df = pd.DataFrame(data)
    df.to_csv("file.csv", index=False)

    with open("file.csv", "rb") as file:
        response = client.post("/uploadcsv/", files={"file": ("file.csv", file, "text/csv")})

    assert response.status_code == 200
    assert "filename" in response.json()
    assert "rows" in response.json()
    assert "columns" in response.json()

    assert response.json()["filename"] == "file.csv"
    assert response.json()["rows"] == df.shape[0]
    assert response.json()["columns"] == df.shape[1]

    os.remove("file.csv")
