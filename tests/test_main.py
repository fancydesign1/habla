from fastapi.testclient import TestClient
from src.main import app


def test_app_creates():
    client = TestClient(app)
    assert client is not None
