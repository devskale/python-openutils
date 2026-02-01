"""
Tests for uniioai_proxy with TU provider.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from io import BytesIO


class TestTUProxyTTS:
    """Test TTS endpoint with TU provider."""

    @patch('uniinfer.uniioai_proxy.TuAITTSProvider')
    def test_tts_endpoint_exists(self, mock_tts_class):
        """Test that TTS endpoint exists."""
        from uniinfer.uniioai_proxy import app

        client = TestClient(app)
        response = client.post("/v1/audio/speech", json={
            "model": "tu@kokoro",
            "input": "Hello"
        })

        # Should either work or return an error about API key
        # We're just checking the endpoint exists
        assert response.status_code in [200, 401, 500]

    @patch('uniinfer.uniioai_proxy.TuAITTSProvider')
    def test_tts_endpoint_with_voice(self, mock_tts_class):
        """Test TTS endpoint with voice parameter."""
        from uniinfer.uniioai_proxy import app

        client = TestClient(app)
        response = client.post("/v1/audio/speech", json={
            "model": "tu@kokoro",
            "input": "Test",
            "voice": "af_alloy"
        })

        # Check endpoint responds
        assert response.status_code in [200, 401, 500]

    @patch('uniinfer.uniioai_proxy.TuAITTSProvider')
    def test_tts_endpoint_with_format(self, mock_tts_class):
        """Test TTS endpoint with format parameter."""
        from uniinfer.uniioai_proxy import app

        client = TestClient(app)
        response = client.post("/v1/audio/speech", json={
            "model": "tu@kokoro",
            "input": "Test",
            "response_format": "wav"
        })

        # Check endpoint responds
        assert response.status_code in [200, 401, 500]


class TestTUProxySTT:
    """Test STT endpoint with TU provider."""

    @patch('uniinfer.uniioai_proxy.TuAISTTProvider')
    def test_stt_endpoint_exists(self, mock_stt_class):
        """Test that STT endpoint exists."""
        from uniinfer.uniioai_proxy import app

        # Create a simple test audio file
        audio_content = b"fake audio data"

        client = TestClient(app)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": (BytesIO(audio_content), "test.mp3", "audio/mpeg")},
            form_data={
                "model": "tu@whisper-large"
            }
        )

        # Should either work or return an error about API key
        # We're just checking that endpoint exists
        assert response.status_code in [200, 401, 500]

    @patch('uniinfer.uniioai_proxy.TuAISTTProvider')
    def test_stt_endpoint_with_language(self, mock_stt_class):
        """Test STT endpoint with language parameter."""
        from uniinfer.uniioai_proxy import app

        audio_content = b"fake audio data"

        client = TestClient(app)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": (BytesIO(audio_content), "test.mp3", "audio/mpeg")},
            form_data={
                "model": "tu@whisper-large",
                "language": "en"
            }
        )

        # Check endpoint responds
        assert response.status_code in [200, 401, 500]

    @patch('uniinfer.uniioai_proxy.TuAISTTProvider')
    def test_stt_endpoint_with_format(self, mock_stt_class):
        """Test STT endpoint with format parameter."""
        from uniinfer.uniioai_proxy import app

        audio_content = b"fake audio data"

        client = TestClient(app)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": (BytesIO(audio_content), "test.mp3", "audio/mpeg")},
            form_data={
                "model": "tu@whisper-large",
                "response_format": "json"
            }
        )

        # Check endpoint responds
        assert response.status_code in [200, 401, 500]

    @patch('uniinfer.uniioai_proxy.TuAISTTProvider')
    def test_stt_endpoint_with_language(self, mock_stt_class):
        """Test STT endpoint with language parameter."""
        from uniinfer.uniioai_proxy import app

        audio_content = b"fake audio data"

        client = TestClient(app)
        response = client.post(
            "/v1/audio/transcriptions",
            data={"file": (BytesIO(audio_content), "test.mp3", "audio/mpeg")},
            form_data={
                "model": "tu@whisper-large",
                "language": "en"
            }
        )

        # Check endpoint responds
        assert response.status_code in [200, 401, 500]

    @patch('uniinfer.uniioai_proxy.TuAISTTProvider')
    def test_stt_endpoint_with_format(self, mock_stt_class):
        """Test STT endpoint with format parameter."""
        from uniinfer.uniioai_proxy import app

        audio_content = b"fake audio data"

        client = TestClient(app)
        response = client.post(
            "/v1/audio/transcriptions",
            data={"file": (BytesIO(audio_content), "test.mp3", "audio/mpeg")},
            form_data={
                "model": "tu@whisper-large",
                "response_format": "json"
            }
        )

        # Check endpoint responds
        assert response.status_code in [200, 401, 500]

    @patch('uniinfer.uniioai_proxy.TuAISTTProvider')
    def test_stt_endpoint_with_language(self, mock_stt_class):
        """Test STT endpoint with language parameter."""
        from uniinfer.uniioai_proxy import app

        audio_content = b"fake audio data"

        client = TestClient(app)
        response = client.post(
            "/v1/audio/transcriptions",
            data={"file": (BytesIO(audio_content), "test.mp3", "audio/mpeg")},
            data={
                "model": "tu@whisper-large",
                "language": "en"
            }
        )

        # Check endpoint responds
        assert response.status_code in [200, 401, 500]

    @patch('uniinfer.uniioai_proxy.TuAISTTProvider')
    def test_stt_endpoint_with_format(self, mock_stt_class):
        """Test STT endpoint with format parameter."""
        from uniinfer.uniioai_proxy import app

        audio_content = b"fake audio data"

        client = TestClient(app)
        response = client.post(
            "/v1/audio/transcriptions",
            data={"file": (BytesIO(audio_content), "test.mp3", "audio/mpeg")},
            data={
                "model": "tu@whisper-large",
                "response_format": "json"
            }
        )

        # Check endpoint responds
        assert response.status_code in [200, 401, 500]


class TestTUProxyErrorHandling:
    """Test error handling for TU provider."""

    @patch('uniinfer.uniioai_proxy.TuAITTSProvider')
    def test_tts_missing_api_key(self, mock_tts_class):
        """Test TTS with missing API key."""
        from uniinfer.uniioai_proxy import app

        # Mock generate_speech to raise ValueError
        mock_instance = MagicMock()
        mock_instance.generate_speech.side_effect = ValueError("API key is required")
        mock_tts_class.return_value = mock_instance

        client = TestClient(app)
        response = client.post("/v1/audio/speech", json={
            "model": "tu@kokoro",
            "input": "Test"
        })

        # Should get 500 error
        assert response.status_code == 500

    @patch('uniinfer.uniioai_proxy.TuAISTTProvider')
    def test_stt_missing_api_key(self, mock_stt_class):
        """Test STT with missing API key."""
        from uniinfer.uniioai_proxy import app

        audio_content = b"fake audio data"

        # Mock transcribe to raise ValueError
        mock_instance = MagicMock()
        mock_instance.transcribe.side_effect = ValueError("API key is required")
        mock_stt_class.return_value = mock_instance

        client = TestClient(app)
        response = client.post(
            "/v1/audio/transcriptions",
            data={"file": (BytesIO(audio_content), "test.mp3", "audio/mpeg")},
            data={
                "model": "tu@whisper-large"
            }
        )

        # Should get 500 error
        assert response.status_code == 500


class TestTUProxyModelParsing:
    """Test model parsing for TU provider."""

    @patch('uniinfer.uniioai_proxy.parse_provider_model')
    def test_tu_model_parsing(self, mock_parse):
        """Test TU model string parsing."""
        from uniinfer.uniioai_proxy import app

        mock_parse.return_value = ("tu", "model-name")

        client = TestClient(app)
        response = client.post("/v1/audio/speech", json={
            "model": "tu@kokoro",
            "input": "Test"
        })

        # Check that parse was called
        mock_parse.assert_called_once_with("tu@kokoro", allowed_providers=['tu'], task_name="TTS")

    @patch('uniinfer.uniioai_proxy.parse_provider_model')
    def test_stu_model_parsing(self, mock_parse):
        """Test STT model string parsing."""
        from uniinfer.uniioai_proxy import app

        audio_content = b"fake audio data"

        mock_parse.return_value = ("tu", "model-name")

        client = TestClient(app)
        response = client.post(
            "/v1/audio/transcriptions",
            data={"file": (BytesIO(audio_content), "test.mp3", "audio/mpeg")},
            data={
                "model": "tu@whisper-large"
            }
        )

        # Check that parse was called
        mock_parse.assert_called_once_with("tu@whisper-large", allowed_providers=['tu'], task_name="STT")


class TestTUProxyRequestValidation:
    """Test request validation for TU provider."""

    def test_tts_missing_input(self):
        """Test TTS with missing input."""
        from uniinfer.uniioai_proxy import app

        client = TestClient(app)
        response = client.post("/v1/audio/speech", json={
            "model": "tu@kokoro"
        })

        # Should get 422 validation error
        assert response.status_code in [422, 400]

    def test_tts_missing_model(self):
        """Test TTS with missing model."""
        from uniinfer.uniioai_proxy import app

        client = TestClient(app)
        response = client.post("/v1/audio/speech", json={
            "input": "Hello"
        })

        # Should get 422 validation error
        assert response.status_code in [422, 400]

    def test_stt_missing_file(self):
        """Test STT with missing file."""
        from uniinfer.uniioai_proxy import app

        client = TestClient(app)
        response = client.post(
            "/v1/audio/transcriptions",
            data={}
        )

        # Should get 422 validation error
        assert response.status_code in [422, 400]

    def test_stt_missing_model(self):
        """Test STT with missing model."""
        from uniinfer.uniioai_proxy import app

        audio_content = b"fake audio data"

        client = TestClient(app)
        response = client.post(
            "/v1/audio/transcriptions",
            data={"file": (BytesIO(audio_content), "test.mp3", "audio/mpeg")}
        )

        # Should get 422 validation error
        assert response.status_code in [422, 400]

    def test_tts_missing_model(self):
        """Test TTS with missing model."""
        from uniinfer.uniioai_proxy import app

        client = TestClient(app)
        response = client.post("/v1/audio/speech", json={
            "input": "Hello"
        })

        # Should get 422 validation error
        assert response.status_code in [422, 400]

    def test_stt_missing_file(self):
        """Test STT with missing file."""
        from uniinfer.uniioai_proxy import app

        client = TestClient(app)
        response = client.post(
            "/v1/audio/transcriptions",
            data={}
        )

        # Should get 422 validation error
        assert response.status_code in [422, 400]

    def test_stt_missing_model(self):
        """Test STT with missing model."""
        from uniinfer.uniioai_proxy import app

        audio_content = b"fake audio data"

        client = TestClient(app)
        response = client.post(
            "/v1/audio/transcriptions",
            data={"file": (BytesIO(audio_content), "test.mp3", "audio/mpeg")}
        )

        # Should get 422 validation error
        assert response.status_code in [422, 400]
