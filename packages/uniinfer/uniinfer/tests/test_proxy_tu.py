"""
Tests for TU TTS/STT proxy endpoints.

Auth: uses PROXY_KEY from .env (credgoo combined token: bearer@encryption).
If PROXY_KEY is not set, tests that need auth are skipped.
"""
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from io import BytesIO

PROXY_KEY = os.getenv("PROXY_KEY")


def _auth_headers():
    if not PROXY_KEY:
        pytest.skip("PROXY_KEY not set in .env")
    return {"Authorization": f"Bearer {PROXY_KEY}"}


# --- TTS endpoint ---

class TestTTSEndpoint:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from uniinfer.proxy_app import app
        return TestClient(app, raise_server_exceptions=False)

    def test_tts_no_auth_returns_401(self, client):
        response = client.post("/v1/audio/speech", json={
            "model": "tu@kokoro", "input": "Hello"
        })
        assert response.status_code == 401

    def test_tts_missing_input_returns_422(self, client):
        response = client.post("/v1/audio/speech", json={
            "model": "tu@kokoro"
        })
        assert response.status_code in [422, 400]

    def test_tts_missing_model_returns_422(self, client):
        response = client.post("/v1/audio/speech", json={
            "input": "Hello"
        })
        assert response.status_code in [422, 400]

    @patch("uniinfer.providers.tu_tts.TuAITTSProvider")
    def test_tts_provider_error_returns_500(self, mock_cls, client):
        mock_instance = MagicMock()
        mock_instance.agenerate_speech = AsyncMock(side_effect=ValueError("API key is required"))
        mock_cls.return_value = mock_instance

        with patch("uniinfer.proxy_routers.media.verify_provider_access", return_value="fake-key"):
            response = client.post("/v1/audio/speech", json={
                "model": "tu@kokoro", "input": "Test"
            })
        assert response.status_code == 500


# --- STT endpoint ---

class TestSTTEndpoint:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from uniinfer.proxy_app import app
        return TestClient(app, raise_server_exceptions=False)

    def test_stt_no_auth_returns_401(self, client):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.mp3", BytesIO(b"audio"), "audio/mpeg")},
            data={"model": "tu@whisper-large"}
        )
        assert response.status_code == 401

    def test_stt_missing_file_returns_422(self, client):
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "tu@whisper-large"}
        )
        assert response.status_code in [422, 400]

    def test_stt_missing_model_returns_422(self, client):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.mp3", BytesIO(b"audio"), "audio/mpeg")}
        )
        assert response.status_code in [422, 400]

    @patch("uniinfer.providers.tu_stt.TuAISTTProvider")
    def test_stt_provider_error_returns_500(self, mock_cls, client):
        mock_instance = MagicMock()
        mock_instance.atranscribe = AsyncMock(side_effect=ValueError("API key is required"))
        mock_cls.return_value = mock_instance

        with patch("uniinfer.proxy_routers.media.verify_provider_access", return_value="fake-key"):
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.mp3", BytesIO(b"audio"), "audio/mpeg")},
                data={"model": "tu@whisper-large"}
            )
        assert response.status_code == 500


# --- Model parsing ---

class TestModelParsing:

    def test_tts_model_parsing(self):
        from uniinfer.proxy_app import parse_provider_model
        provider, model = parse_provider_model("tu@kokoro")
        assert provider == "tu"
        assert model == "kokoro"

    def test_stt_model_parsing(self):
        from uniinfer.proxy_app import parse_provider_model
        provider, model = parse_provider_model("tu@whisper-large")
        assert provider == "tu"
        assert model == "whisper-large"
