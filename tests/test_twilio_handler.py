import os
os.environ.setdefault("PUBLIC_URL", "test.example.com")

import pytest
from fastapi.testclient import TestClient
from src.main import app


def test_twiml_endpoint_returns_stream_connect():
    client = TestClient(app)
    response = client.post("/twiml", data={"CallSid": "CA123"})
    assert response.status_code == 200
    assert "Connect" in response.text
    assert "Stream" in response.text
    assert "/ws/call" in response.text


def test_twiml_uses_wss_scheme():
    client = TestClient(app)
    response = client.post("/twiml", data={"CallSid": "CA123"})
    assert "wss://" in response.text
