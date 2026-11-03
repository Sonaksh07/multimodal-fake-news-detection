import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from database.db import get_db, SessionLocal
from database.models import PredictionRecord

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "Multimodal" in data["system"]


def test_predict_text_endpoint():
    payload = {
        "text": "Government announces major investment in clean renewable solar infrastructure.",
        "title": "Clean Energy Announcement"
    }
    response = client.post("/api/v1/predict/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "text_bert_score" in data
    assert "text_tfidf_score" in data
    assert "fusion_result" in data
    assert data["fusion_result"]["final_label"] in ["Likely Fake", "Likely Authentic", "Uncertain"]


def test_predict_text_empty_error():
    response = client.post("/api/v1/predict/text", json={"text": "   "})
    assert response.status_code == 400


def test_predict_multimodal_text_only():
    form_data = {
        "text": "Secret whistleblowers claim alien base discovered on moon.",
        "title": "Moon Alien Story"
    }
    response = client.post("/api/v1/predict/multimodal", data=form_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert "fusion_result" in data
    assert data["fusion_result"]["final_label"] in ["Likely Fake", "Likely Authentic", "Uncertain"]


def test_dataset_endpoint():
    response = client.get("/api/v1/dataset?name=demo")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset_name"] == "demo"
    assert data["total_records"] > 0


def test_analytics_comparison_endpoint():
    response = client.get("/api/v1/analytics/comparison?sample_size=5")
    assert response.status_code == 200
    data = response.json()
    assert "nlp_models_comparison" in data
    assert "modality_fusion_comparison" in data


def test_report_generation_endpoint():
    # Submit prediction first to generate record
    form_data = {
        "text": "Official central bank report indicates stable inflation metrics.",
        "title": "Inflation Report"
    }
    resp = client.post("/api/v1/predict/multimodal", data=form_data)
    assert resp.status_code == 200
    pred_id = resp.json()["prediction_id"]

    # Generate PDF report
    rep_resp = client.post(f"/api/v1/reports/generate?prediction_id={pred_id}")
    assert rep_resp.status_code == 200
    rep_data = rep_resp.json()
    assert rep_data["status"] == "success"
    assert os.path.exists(rep_data["pdf_path"])
