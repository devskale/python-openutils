"""
Simple validation tests for TU TTS/STT async implementations.
"""
from uniinfer.providers.tu_tts import TuAITTSProvider
from uniinfer.providers.tu_stt import TuAISTTProvider
from uniinfer import TTSRequest, TTSResponse, STTRequest, STTResponse
import inspect


class TestTuTTSProviderAsync:
    """Test TU TTS provider async methods."""

    def test_agenerate_speech_exists(self):
        """Test that agenerate_speech method exists."""
        provider = TuAITTSProvider(api_key="test-key")
        assert hasattr(provider, 'agenerate_speech')
        assert callable(provider.agenerate_speech)

    def test_agenerate_speech_is_coroutine(self):
        """Test that agenerate_speech is a coroutine function."""
        assert inspect.iscoroutinefunction(TuAITTSProvider.agenerate_speech)

    def test_generate_speech_exists(self):
        """Test that generate_speech method exists."""
        provider = TuAITTSProvider(api_key="test-key")
        assert hasattr(provider, 'generate_speech')
        assert callable(provider.generate_speech)

    def test_generate_speech_is_function(self):
        """Test that generate_speech is a regular function."""
        assert inspect.isfunction(TuAITTSProvider.generate_speech)

    def test_list_models_exists(self):
        """Test that list_models method exists."""
        provider = TuAITTSProvider(api_key="test-key")
        assert hasattr(provider, 'list_models')
        assert callable(provider.list_models)

    def test_list_models_is_classmethod(self):
        """Test that list_models is a class method."""
        assert inspect.ismethod(TuAITTSProvider.list_models)

    def test_tts_request_creation(self):
        """Test TTSRequest creation."""
        request = TTSRequest(
            input="Hello world",
            model="kokoro",
            voice="af_alloy",
            response_format="mp3",
            speed=1.5,
            instructions="Speak clearly"
        )

        assert request.input == "Hello world"
        assert request.model == "kokoro"
        assert request.voice == "af_alloy"
        assert request.response_format == "mp3"
        assert request.speed == 1.5
        assert request.instructions == "Speak clearly"

    def test_tts_response_creation(self):
        """Test TTSResponse creation."""
        response = TTSResponse(
            audio_content=b"fake audio",
            model="kokoro",
            provider="tu",
            content_type="audio/mpeg",
            raw_response={}
        )

        assert response.audio_content == b"fake audio"
        assert response.model == "kokoro"
        assert response.provider == "tu"
        assert response.content_type == "audio/mpeg"


class TestTuSTTProviderAsync:
    """Test TU STT provider async methods."""

    def test_atranscribe_exists(self):
        """Test that atranscribe method exists."""
        provider = TuAISTTProvider(api_key="test-key")
        assert hasattr(provider, 'atranscribe')
        assert callable(provider.atranscribe)

    def test_atranscribe_is_coroutine(self):
        """Test that atranscribe is a coroutine function."""
        assert inspect.iscoroutinefunction(TuAISTTProvider.atranscribe)

    def test_transcribe_exists(self):
        """Test that transcribe method exists."""
        provider = TuAISTTProvider(api_key="test-key")
        assert hasattr(provider, 'transcribe')
        assert callable(provider.transcribe)

    def test_transcribe_is_function(self):
        """Test that transcribe is a regular function."""
        assert inspect.isfunction(TuAISTTProvider.transcribe)

    def test_list_models_exists(self):
        """Test that list_models method exists."""
        provider = TuAISTTProvider(api_key="test-key")
        assert hasattr(provider, 'list_models')
        assert callable(provider.list_models)

    def test_list_models_is_classmethod(self):
        """Test that list_models is a class method."""
        assert inspect.ismethod(TuAISTTProvider.list_models)

    def test_stt_request_creation(self):
        """Test STTRequest creation."""
        request = STTRequest(
            file=b"fake audio data",
            model="whisper-large",
            language="en",
            response_format="json",
            temperature=0.1,
            prompt="Transcribe this",
            timestamp_granularities=["word"]
        )

        assert request.file == b"fake audio data"
        assert request.model == "whisper-large"
        assert request.language == "en"
        assert request.response_format == "json"
        assert request.temperature == 0.1
        assert request.prompt == "Transcribe this"
        assert request.timestamp_granularities == ["word"]

    def test_stt_response_creation(self):
        """Test STTResponse creation."""
        response = STTResponse(
            text="Transcribed text",
            model="whisper-large",
            provider="tu",
            language="en",
            duration=2.5,
            segments=[
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello"
                }
            ],
            raw_response={}
        )

        assert response.text == "Transcribed text"
        assert response.model == "whisper-large"
        assert response.provider == "tu"
        assert response.language == "en"
        assert response.duration == 2.5
        assert len(response.segments) == 1


class TestTUProviderInheritance:
    """Test TU provider inheritance."""

    def test_tts_provider_inherits_base(self):
        """Test TuAITTSProvider inherits from TTSProvider."""
        from uniinfer.core import TTSProvider
        assert issubclass(TuAITTSProvider, TTSProvider)

    def test_stt_provider_inherits_base(self):
        """Test TuAISTTProvider inherits from STTProvider."""
        from uniinfer.core import STTProvider
        assert issubclass(TuAISTTProvider, STTProvider)


class TestTUProviderAttributes:
    """Test TU provider attributes."""

    def test_tts_base_url(self):
        """Test TTS base URL."""
        assert hasattr(TuAITTSProvider, 'BASE_URL')
        assert TuAITTSProvider.BASE_URL == "https://aqueduct.ai.datalab.tuwien.ac.at/v1"

    def test_stt_base_url(self):
        """Test STT base URL."""
        assert hasattr(TuAISTTProvider, 'BASE_URL')
        assert TuAISTTProvider.BASE_URL == "https://aqueduct.ai.datalab.tuwien.ac.at/v1"

    def test_tts_init(self):
        """Test TTS provider initialization."""
        provider = TuAITTSProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.organization is None

    def test_tts_init_with_organization(self):
        """Test TTS provider initialization with organization."""
        provider = TuAITTSProvider(api_key="test-key", organization="test-org")
        assert provider.api_key == "test-key"
        assert provider.organization == "test-org"

    def test_stt_init(self):
        """Test STT provider initialization."""
        provider = TuAISTTProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.organization is None

    def test_stt_init_with_organization(self):
        """Test STT provider initialization with organization."""
        provider = TuAISTTProvider(api_key="test-key", organization="test-org")
        assert provider.api_key == "test-key"
        assert provider.organization == "test-org"
