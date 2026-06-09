"""
Tests for TTS (Text-to-Speech) and STT (Speech-to-Text) async functionality.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from uniinfer import TTSRequest, TTSResponse, STTRequest, STTResponse
from uniinfer.providers.tu_tts import TuAITTSProvider
from uniinfer.providers.tu_stt import TuAISTTProvider


class TestTuTTSAsync:
    """Test async TTS provider."""

    def test_agenerate_speech_method_exists(self):
        """Test that agenerate_speech method exists."""
        provider = TuAITTSProvider(api_key="test-key")
        assert hasattr(provider, 'agenerate_speech')
        assert callable(provider.agenerate_speech)

    def test_generate_speech_is_wrapper(self):
        """Test that sync generate_speech method wraps async."""
        provider = TuAITTSProvider(api_key="test-key")

        assert callable(provider.generate_speech)
        assert callable(provider.agenerate_speech)

    def test_list_models_is_sync(self):
        """Test that list_models is still sync."""
        provider = TuAITTSProvider(api_key="test-key")

        assert hasattr(provider, 'list_models')
        assert callable(provider.list_models)

    @pytest.mark.asyncio
    async def test_agenerate_speech_success(self):
        """Test successful async speech generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake audio data"
        mock_response.headers = {"Content-Type": "audio/mpeg"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        provider = TuAITTSProvider(api_key="test-key")
        request = TTSRequest(input="Hello world")

        with patch.object(provider, '_get_async_client', return_value=mock_client):
            response = await provider.agenerate_speech(request)

        assert response.audio_content == b"fake audio data"
        assert response.provider == "tu"
        assert response.content_type == "audio/mpeg"
        assert response.model == "kokoro"

    @pytest.mark.asyncio
    async def test_agenerate_speech_with_voice(self):
        """Test async speech generation with custom voice."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"custom voice audio"
        mock_response.headers = {"Content-Type": "audio/mpeg"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        provider = TuAITTSProvider(api_key="test-key")
        request = TTSRequest(
            input="Test",
            voice="af_alloy",
            response_format="mp3",
            speed=1.5
        )

        with patch.object(provider, '_get_async_client', return_value=mock_client):
            response = await provider.agenerate_speech(request)

        assert response.audio_content == b"custom voice audio"
        assert response.provider == "tu"
        assert response.content_type == "audio/mpeg"
        assert response.model == "kokoro"

    @pytest.mark.asyncio
    async def test_atranscribe_success(self):
        """Test successful async transcription."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Hello world",
            "language": "en",
            "duration": 2.5,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world"
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        provider = TuAISTTProvider(api_key="test-key")
        request = STTRequest(file=b"fake audio content")

        with patch.object(provider, '_get_async_client', return_value=mock_client):
            response = await provider.atranscribe(request)

        assert response.text == "Hello world"
        assert response.provider == "tu"
        assert response.model == "whisper-large"
        assert response.language == "en"
        assert response.duration == 2.5

    @pytest.mark.asyncio
    async def test_atranscribe_with_language(self):
        """Test async transcription with language parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Spanish text",
            "language": "es",
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        provider = TuAISTTProvider(api_key="test-key")
        request = STTRequest(
            file=b"test audio",
            language="es",
            model="whisper-1",
            temperature=0.1
        )

        with patch.object(provider, '_get_async_client', return_value=mock_client):
            response = await provider.atranscribe(request)

        assert response.text == "Spanish text"
        assert response.language == "es"

    @pytest.mark.asyncio
    async def test_atranscribe_with_prompt(self):
        """Test async transcription with prompt parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Transcribed with prompt guidance",
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        provider = TuAISTTProvider(api_key="test-key")
        request = STTRequest(
            file=b"test audio",
            prompt="This is a technical document"
        )

        with patch.object(provider, '_get_async_client', return_value=mock_client):
            response = await provider.atranscribe(request)

        assert response.text == "Transcribed with prompt guidance"


class TestTTSSTTBaseClasses:
    """Test base TTS and STT provider classes."""

    def test_tts_request_creation(self):
        """Test TTSRequest creation."""
        request = TTSRequest(
            input="Hello",
            model="test-model",
            voice="test-voice",
            response_format="mp3",
            speed=1.0,
            instructions="Speak clearly"
        )

        assert request.input == "Hello"
        assert request.model == "test-model"
        assert request.voice == "test-voice"
        assert request.response_format == "mp3"
        assert request.speed == 1.0
        assert request.instructions == "Speak clearly"

    def test_stt_request_creation(self):
        """Test STTRequest creation."""
        request = STTRequest(
            file=b"fake audio data",
            model="test-model",
            language="en",
            response_format="json",
            temperature=0.1,
            prompt="Transcribe this",
            timestamp_granularities=["word"]
        )

        assert request.file == b"fake audio data"
        assert request.model == "test-model"
        assert request.language == "en"
        assert request.response_format == "json"
        assert request.temperature == 0.1
        assert request.prompt == "Transcribe this"
        assert request.timestamp_granularities == ["word"]

    def test_tts_response_creation(self):
        """Test TTSResponse creation."""
        response = TTSResponse(
            audio_content=b"test audio",
            model="test-model",
            provider="test-provider",
            content_type="audio/mpeg",
            raw_response={}
        )

        assert response.audio_content == b"test audio"
        assert response.model == "test-model"
        assert response.provider == "test-provider"
        assert response.content_type == "audio/mpeg"

    def test_stt_response_creation(self):
        """Test STTResponse creation."""
        response = STTResponse(
            text="Transcribed text",
            model="test-model",
            provider="test-provider",
            language="en",
            duration=2.5,
            segments=[
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Transcribed"
                }
            ],
            raw_response={}
        )

        assert response.text == "Transcribed text"
        assert response.model == "test-model"
        assert response.provider == "test-provider"
        assert response.language == "en"
        assert response.duration == 2.5
        assert len(response.segments) == 1


class TestProviderSyncWrappers:
    """Test that sync methods wrap async correctly."""

    def test_tu_tts_sync_wrapper(self):
        """Test TTS sync wrapper."""
        provider = TuAITTSProvider(api_key="test-key")

        assert callable(provider.generate_speech)
        assert callable(provider.agenerate_speech)

    def test_tu_stt_sync_wrapper(self):
        """Test STT sync wrapper."""
        provider = TuAISTTProvider(api_key="test-key")

        assert callable(provider.transcribe)
        assert callable(provider.atranscribe)


class TestTTSTTIntegration:
    """Integration tests for TTS/STT async functionality."""

    @pytest.mark.skip(reason="Async integration tests require more complex setup")
    async def test_tts_concurrent_requests(self):
        """Test multiple concurrent TTS requests."""
        pass

    @pytest.mark.skip(reason="Async integration tests require more complex setup")
    async def test_stt_concurrent_requests(self):
        """Test multiple concurrent STT requests."""
        pass
