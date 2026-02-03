"""
Integration tests for TTS and STT with live API calls.
Requires actual API credentials in .env file.
"""
import asyncio  # noqa: F401
import os
import tempfile
import wave
import pytest

from credgoo.credgoo import get_api_key
from uniinfer import TTSRequest, STTRequest
from uniinfer.providers.tu_tts import TuAITTSProvider
from uniinfer.providers.tu_stt import TuAISTTProvider


def get_api_key_cached(service: str) -> str | None:
    """Get API key from credgoo or env var."""
    key = get_api_key(service)
    if key:
        return key
    env_key = os.getenv(f"{service.upper()}_API_KEY")
    return env_key


class TestTTSIntegration:
    """Integration tests for TTS with live TU API."""

    @pytest.fixture
    def api_key(self):
        """Get API key for testing."""
        key = get_api_key_cached("tu")
        pytest.skip("No TU API key available") if not key else key
        return key

    @pytest.fixture
    def tts_provider(self, api_key):
        """Create TTS provider instance."""
        return TuAITTSProvider(api_key=api_key)

    def test_api_key_retrieval(self, api_key):
        """Test that API key can be retrieved."""
        assert api_key is not None
        assert len(api_key) > 10

    @pytest.mark.asyncio
    async def test_tts_kokoro_af_heart(self, tts_provider):
        """Test TTS with kokoro model and af_heart voice."""
        request = TTSRequest(
            input="Hello, this is a test of the text to speech system.",
            model="kokoro",
            voice="af_heart"
        )

        response = await tts_provider.agenerate_speech(request)

        assert response is not None
        assert response.audio_content is not None
        assert len(response.audio_content) > 0
        assert response.provider == "tu"
        assert response.model == "kokoro"
        assert response.content_type in ["audio/mpeg", "audio/mp3"]

    @pytest.mark.asyncio
    async def test_tts_kokoro_af_bella(self, tts_provider):
        """Test TTS with kokoro model and af_bella voice."""
        request = TTSRequest(
            input="Testing another voice.",
            model="kokoro",
            voice="af_bella"
        )

        response = await tts_provider.agenerate_speech(request)

        assert response is not None
        assert response.audio_content is not None
        assert len(response.audio_content) > 0

    @pytest.mark.asyncio
    async def test_tts_kokoro_af_nicole(self, tts_provider):
        """Test TTS with kokoro model and af_nicole voice."""
        request = TTSRequest(
            input="Testing the Nicole voice.",
            model="kokoro",
            voice="af_nicole"
        )

        response = await tts_provider.agenerate_speech(request)

        assert response is not None
        assert response.audio_content is not None
        assert len(response.audio_content) > 0

    @pytest.mark.asyncio
    async def test_tts_with_speed(self, tts_provider):
        """Test TTS with custom speed."""
        request = TTSRequest(
            input="Testing faster speech.",
            model="kokoro",
            voice="af_heart",
            speed=1.5
        )

        response = await tts_provider.agenerate_speech(request)

        assert response is not None
        assert response.audio_content is not None

    @pytest.mark.asyncio
    async def test_tts_thorsten_voice(self, tts_provider):
        """Test TTS with thorsten voice (if available)."""
        request = TTSRequest(
            input="Testing German voice.",
            model="kokoro",
            voice="thorsten"
        )

        response = await tts_provider.agenerate_speech(request)

        assert response is not None
        assert response.audio_content is not None
        assert len(response.audio_content) > 0

    @pytest.mark.asyncio
    async def test_tts_sync_wrapper(self, tts_provider):
        """Test sync wrapper method."""
        request = TTSRequest(
            input="Testing sync wrapper.",
            model="kokoro",
            voice="af_heart"
        )

        response = tts_provider.generate_speech(request)

        assert response is not None
        assert response.audio_content is not None
        assert len(response.audio_content) > 0


class TestSTTIntegration:
    """Integration tests for STT with live TU API."""

    @pytest.fixture
    def api_key(self):
        """Get API key for testing."""
        key = get_api_key_cached("tu")
        pytest.skip("No TU API key available") if not key else key
        return key

    @pytest.fixture
    def stt_provider(self, api_key):
        """Create STT provider instance."""
        return TuAISTTProvider(api_key=api_key)

    @pytest.fixture
    def sample_audio_file(self):
        """Create a temporary audio file for testing."""
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Create a simple WAV file (silent, 1 second)
            num_channels = 1
            sample_width = 2
            sample_rate = 16000
            num_frames = sample_rate  # 1 second

            with wave.open(f.name, 'w') as wav_file:
                wav_file.setparams((num_channels, sample_width, sample_rate, num_frames, 'NONE', 'not compressed'))
                wav_file.writeframes(b'\x00' * (num_frames * sample_width))

            yield f.name

            os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_stt_whisper_large(self, stt_provider, sample_audio_file):
        """Test STT with whisper-large model."""
        request = STTRequest(
            file=sample_audio_file,
            model="whisper-large"
        )

        response = await stt_provider.atranscribe(request)

        assert response is not None
        assert response.text is not None
        assert response.provider == "tu"
        assert response.model == "whisper-large"

    @pytest.mark.asyncio
    async def test_stt_with_language(self, stt_provider, sample_audio_file):
        """Test STT with language parameter."""
        request = STTRequest(
            file=sample_audio_file,
            model="whisper-large",
            language="en"
        )

        response = await stt_provider.atranscribe(request)

        assert response is not None
        assert response.provider == "tu"

    @pytest.mark.asyncio
    async def test_stt_sync_wrapper(self, stt_provider, sample_audio_file):
        """Test sync wrapper method."""
        request = STTRequest(
            file=sample_audio_file,
            model="whisper-large"
        )

        response = stt_provider.transcribe(request)

        assert response is not None
        assert response.text is not None
        assert response.provider == "tu"

    @pytest.mark.asyncio
    async def test_stt_with_bytes_input(self, stt_provider):
        """Test STT with bytes input instead of file path."""
        # Create simple audio bytes
        import struct

        sample_rate = 16000
        duration = 0.5  # 0.5 seconds
        num_frames = int(sample_rate * duration)

        # Create simple WAV header + data
        import io

        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'w') as wav_file:
                wav_file.setparams((1, 2, sample_rate, num_frames, 'NONE', 'not compressed'))
                wav_file.writeframes(struct.pack('h', 0) * num_frames)

            audio_bytes = wav_buffer.getvalue()

        request = STTRequest(
            file=audio_bytes,
            model="whisper-large"
        )

        response = await stt_provider.atranscribe(request)

        assert response is not None
        assert response.provider == "tu"


class TestTTSErrorHandling:
    """Test error handling for TTS/STT."""

    @pytest.mark.asyncio
    async def test_tts_missing_api_key(self):
        """Test that error is raised when API key is missing."""
        provider = TuAITTSProvider(api_key=None)

        request = TTSRequest(
            input="Test",
            model="kokoro"
        )

        with pytest.raises(ValueError, match="API key is required"):
            await provider.agenerate_speech(request)

    @pytest.mark.asyncio
    async def test_stt_missing_api_key(self):
        """Test that error is raised when API key is missing."""
        provider = TuAISTTProvider(api_key=None)

        request = STTRequest(
            file=b"test audio",
            model="whisper-large"
        )

        with pytest.raises(ValueError, match="API key is required"):
            await provider.atranscribe(request)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
