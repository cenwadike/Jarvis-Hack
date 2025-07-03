import pytest
import numpy as np
import threading
import io
import wave
import logging
from unittest.mock import MagicMock, Mock, patch
from contextlib import AbstractContextManager, contextmanager
from freezegun import freeze_time
from src.voice.voice_parser import VoiceCommandConfig, LocalAudioProcessor, OfflineVoiceRecognizer, VoiceCommandValidator, OfflineVoiceParser, VoiceParsingError
from types import SimpleNamespace

# Test setup
@pytest.fixture
def voice_config():
    """Provide a VoiceCommandConfig instance for tests."""
    return VoiceCommandConfig(
        max_audio_duration=5,
        max_cpu=4,
        min_cpu=1,
        max_memory_gb=8,
        cache_ttl=60,
        max_audio_size_mb=2,
        sample_rate=16000,
        channels=1
    )

@pytest.fixture
def valid_audio():
    """Generate a valid WAV audio for testing."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data.tobytes())
    audio_buffer.seek(0)
    return audio_buffer.read()

@pytest.fixture
def mock_spacy(mocker):
    """Mock spaCy model loading and processing."""
    mock_nlp = Mock()
    mock_doc = Mock()
    mock_doc.__iter__.return_value = [
        Mock(text="deploy", is_digit=False),
        Mock(text="nginx", is_digit=False),
        Mock(text="with", is_digit=False),
        Mock(text="2", is_digit=True),
        Mock(text="cpu", is_digit=False),
        Mock(text="and", is_digit=False),
        Mock(text="1", is_digit=True),
        Mock(text="Gi", is_digit=False)
    ]
    mock_nlp.return_value = mock_doc
    mocker.patch('spacy.load', return_value=mock_nlp)
    return mock_nlp

@pytest.fixture
def mock_vosk(mocker):
    """Mock Vosk model and recognizer."""
    mock_model = Mock()
    mock_recognizer = Mock()
    mock_recognizer.AcceptWaveform = Mock()
    mock_recognizer.FinalResult = Mock(return_value='{"text": "deploy nginx with 2 cpu and 1 Gi"}')
    mocker.patch('vosk.Model', return_value=mock_model)
    mocker.patch('vosk.KaldiRecognizer', return_value=mock_recognizer)
    return mock_recognizer

@pytest.fixture
def caplog(caplog):
    """Capture log messages."""
    caplog.set_level(logging.DEBUG)
    return caplog

def test_local_audio_processor_singleton(voice_config, caplog):
    """Test that LocalAudioProcessor is a singleton."""
    processor1 = LocalAudioProcessor(voice_config)
    processor2 = LocalAudioProcessor(voice_config)
    assert processor1 is processor2
    assert "LocalAudioProcessor instantiated" not in caplog.text  # Singleton prevents re-instantiation

@patch('pyaudio.PyAudio')
def test_local_audio_processor_process_valid_audio(mock_pyaudio, voice_config, valid_audio, caplog):
    """Test processing valid audio bytes."""
    processor = LocalAudioProcessor(voice_config)
    result = processor.process_audio_from_bytes(valid_audio)
    assert isinstance(result, np.ndarray)
    assert len(result) > 0
    assert f"Processed {len(result)/voice_config.sample_rate:.2f} seconds of audio" in caplog.text

@patch('pyaudio.PyAudio')
def test_local_audio_processor_invalid_audio_format(mock_pyaudio, voice_config, caplog):
    """Test processing audio with invalid format."""
    processor = LocalAudioProcessor(voice_config)
    result = processor.process_audio_from_bytes(b"not a wav file")
    assert result is None
    assert "Audio processing failed: file does not start with RIFF id" in caplog.text

@patch('pyaudio.PyAudio')
def test_local_audio_processor_invalid_channels(mock_pyaudio, voice_config, caplog):
    """Test processing audio with wrong number of channels."""
    invalid_audio = io.BytesIO()
    with wave.open(invalid_audio, 'wb') as wav:
        wav.setnchannels(2)  # Stereo instead of mono
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
    invalid_audio.seek(0)
    processor = LocalAudioProcessor(voice_config)
    result = processor.process_audio_from_bytes(invalid_audio.read())
    assert result is None
    assert "Audio processing failed: Audio must be mono" in caplog.text

@patch('pyaudio.PyAudio')
def test_local_audio_processor_too_large_audio(mock_pyaudio, voice_config, caplog):
    """Test processing audio exceeding size limit."""
    large_audio = io.BytesIO()
    with wave.open(large_audio, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros(3 * 1024 * 1024 // 2, dtype=np.int16).tobytes())  # >2MB
    large_audio.seek(0)
    processor = LocalAudioProcessor(voice_config)
    result = processor.process_audio_from_bytes(large_audio.read())
    assert result is None
    assert f"Audio processing failed: Audio size exceeds {voice_config.max_audio_size_mb}MB limit" in caplog.text
    
@patch('pyaudio.PyAudio')
def test_local_audio_processor_resource_cleanup(mock_pyaudio, voice_config, caplog):
    """Test proper resource cleanup."""
    LocalAudioProcessor._instances.clear()
    mock_audio_instance = mock_pyaudio.return_value

    processor = LocalAudioProcessor(voice_config)
    with processor:
        assert processor._initialized
    assert not processor._initialized

    mock_audio_instance.terminate.assert_called_once()

from unittest.mock import patch, Mock
    
@patch('src.voice.voice_parser.spacy.load')
@patch('src.voice.voice_parser.Model')
@patch('src.voice.voice_parser.KaldiRecognizer')
def test_offline_voice_recognizer_success(mock_kaldi, mock_model, mock_spacy, voice_config, caplog):
    """Test successful speech recognition."""
    # Setup mock recognizer
    mock_recognizer = Mock()
    mock_recognizer.AcceptWaveform.return_value = True
    mock_recognizer.FinalResult.return_value = '{"text": "deploy nginx"}'
    mock_kaldi.return_value = mock_recognizer

    # Instantiate recognizer (uses mocked KaldiRecognizer + Model)
    recognizer = OfflineVoiceRecognizer(voice_config)

    # Provide sample audio input
    audio_data = np.ones(16000, dtype=np.float32)  # 1 second of dummy audio
    text = recognizer.recognize_speech(audio_data)

    # Validate results
    assert text == "deploy nginx"
    assert "Recognized text: deploy nginx" in caplog.text

@patch('src.voice.voice_parser.spacy.load')
@patch('src.voice.voice_parser.Model')
@patch('src.voice.voice_parser.KaldiRecognizer')
def test_offline_voice_recognizer_empty_audio(mock_kaldi, mock_model, mock_spacy, voice_config, caplog):
    """Test speech recognition with empty audio."""
    recognizer = OfflineVoiceRecognizer(voice_config)
    text = recognizer.recognize_speech(np.array([]))
    assert text == ""
    assert "Recognized text" not in caplog.text

@patch('src.voice.voice_parser.spacy.load')
@patch('src.voice.voice_parser.Model')
@patch('src.voice.voice_parser.KaldiRecognizer')
def test_offline_voice_recognizer_vosk_failure(mock_kaldi, mock_model, mock_spacy, voice_config, caplog):
    """Test speech recognition failure due to Vosk error."""
    mock_recognizer = Mock()
    mock_recognizer.AcceptWaveform = Mock(side_effect=Exception("Vosk error"))
    mock_kaldi.return_value = mock_recognizer
    recognizer = OfflineVoiceRecognizer(voice_config)
    text = recognizer.recognize_speech(np.ones(16000, dtype=np.float32))
    assert text == ""
    assert "Speech recognition failed: Vosk error" in caplog.text

@patch('spacy.load', side_effect=OSError("Model not found"))
def test_offline_voice_recognizer_spacy_failure(mock_spacy, voice_config):
    """Test initialization failure due to missing spaCy model."""
    with pytest.raises(VoiceParsingError, match="NLP model not available"):
        OfflineVoiceRecognizer(voice_config)

def test_voice_command_validator_deployment_id(voice_config):
    """Test deployment ID validation."""
    validator = VoiceCommandValidator(voice_config)
    assert validator.validate_deployment_id("test-123")
    assert not validator.validate_deployment_id("invalid@id")
    assert not validator.validate_deployment_id("a" * 51)  # Too long

def test_voice_command_validator_image_name(voice_config, caplog):
    """Test image name validation."""
    validator = VoiceCommandValidator(voice_config)
    assert validator.validate_image_name("nginx")
    assert not validator.validate_image_name("invalid_image")
    assert "Invalid image name: invalid_image" not in caplog.text  # Validation is silent

def test_voice_command_validator_cpu(voice_config):
    """Test CPU validation."""
    validator = VoiceCommandValidator(voice_config)
    assert validator.validate_cpu(2)
    assert not validator.validate_cpu(5)  # Exceeds max_cpu=4
    assert not validator.validate_cpu(-1)

def test_voice_command_validator_memory(voice_config, caplog):
    """Test memory validation."""
    validator = VoiceCommandValidator(voice_config)
    assert validator.validate_memory("1Gi")
    assert not validator.validate_memory("10Gi")  # Exceeds max_memory_gb=8
    assert not validator.validate_memory("invalid")
    assert "Invalid memory value" not in caplog.text  # Validation is silent

def test_voice_command_validator_storage(voice_config):
    """Test storage validation."""
    validator = VoiceCommandValidator(voice_config)
    assert validator.validate_storage("512Mi")
    assert not validator.validate_storage("1001Gi")  # Exceeds max_storage_gb=1000
    assert not validator.validate_storage("invalid")

def test_voice_command_validator_ports(voice_config):
    """Test ports validation."""
    validator = VoiceCommandValidator(voice_config)
    assert validator.validate_ports(["80", "443"])
    assert not validator.validate_ports(["invalid"])
    assert not validator.validate_ports(["80"] * 11)  # Exceeds max_ports=10

def test_voice_command_validator_sanitize_text(voice_config, caplog):
    """Test text sanitization."""
    validator = VoiceCommandValidator(voice_config)
    sanitized = validator.sanitize_text("deploy nginx \u202E malicious")
    assert sanitized == "deploy nginx  malicious"
    assert len(validator.sanitize_text("a" * 1000)) == voice_config.max_text_length
    assert "Sanitized text" not in caplog.text  # Sanitization is silent

@patch('src.voice.voice_parser.LocalAudioProcessor.process_audio_from_bytes')
@patch('src.voice.voice_parser.OfflineVoiceRecognizer')
def test_offline_voice_parser_success(mock_recognizer_class, mock_process_audio, voice_config, valid_audio, caplog):
    """Test successful voice command parsing from audio with full dependency mocking."""

    # 1. Simulate preprocessed audio array
    mock_process_audio.return_value = np.ones(16000, dtype=np.float32)

    # 2. Mock recognizer and its speech + NLP pipeline
    mock_recognizer = Mock()
    mock_recognizer.recognize_speech.return_value = "deploy nginx with 2 cpu and 1 gi"

    # 3. Use real token-like objects (SimpleNamespace)
    tokens = [
        SimpleNamespace(text="deploy", is_digit=False),
        SimpleNamespace(text="nginx", is_digit=False),
        SimpleNamespace(text="service", is_digit=False),  # triggers target → deployment
        SimpleNamespace(text="with", is_digit=False),
        SimpleNamespace(text="2", is_digit=True),
        SimpleNamespace(text="cpu", is_digit=False),
        SimpleNamespace(text="and", is_digit=False),
        SimpleNamespace(text="1", is_digit=True),
        SimpleNamespace(text="Gi", is_digit=False)
    ]

    # 4. Connect recognizer.nlp to return the mock token sequence
    mock_recognizer.nlp.return_value = tokens
    mock_recognizer_class.return_value = mock_recognizer

    # 5. Parse audio with test subject
    parser = OfflineVoiceParser(voice_config)
    command, text = parser.parse_voice_command_from_audio(valid_audio)

    # 6. Assertions
    assert text == "deploy nginx with 2 cpu and 1 gi"
    assert command["action"] == "deploy"
    assert command["target"] == "deployment"
    assert command["image"] == "nginx"
    assert command["cpu"] == 2.0
    assert command["memory"] == "1Gi"
    assert command["storage"] == "512Mi"
    assert command["ports"] == ["80"]
    assert "Successfully parsed voice command" in caplog.text
          
@patch('pyaudio.PyAudio')
@patch('vosk.Model')
@patch('vosk.KaldiRecognizer')
@patch('spacy.load')
def test_offline_voice_parser_invalid_audio(mock_spacy, mock_kaldi, mock_model, mock_pyaudio, voice_config, caplog):
    """Test parsing with invalid audio."""
    parser = OfflineVoiceParser(voice_config)
    command, text = parser.parse_voice_command_from_audio(b"invalid audio")
    assert command["action"] is None
    assert text == "No valid audio data"
    assert "Audio processing failed: file does not start with RIFF id" in caplog.text

@patch('pyaudio.PyAudio')
@patch('vosk.Model')
@patch('vosk.KaldiRecognizer')
@patch('spacy.load')
def test_offline_voice_parser_empty_text(mock_spacy, mock_kaldi, mock_model, mock_pyaudio, voice_config, valid_audio, caplog):
    """Test parsing with empty recognized text."""
    mock_recognizer = Mock()
    mock_recognizer.AcceptWaveform = Mock()
    mock_recognizer.FinalResult = Mock(return_value='{"text": ""}')
    mock_kaldi.return_value = mock_recognizer
    parser = OfflineVoiceParser(voice_config)
    command, text = parser.parse_voice_command_from_audio(valid_audio)
    assert command["action"] is None
    assert text == "Could not understand audio"
    assert "Recognized text" not in caplog.text

@patch('src.voice.voice_parser.LocalAudioProcessor.process_audio_from_bytes')
@patch('src.voice.voice_parser.OfflineVoiceRecognizer')
def test_offline_voice_parser_invalid_command(mock_recognizer_class, mock_process_audio, voice_config, caplog):
    """Test parsing invalid command structure."""

    # 1. Simulate successful audio preprocessing
    mock_process_audio.return_value = np.ones(16000, dtype=np.float32)

    # 2. Mock recognizer and NLP
    mock_recognizer = Mock()
    mock_recognizer.recognize_speech.return_value = "terminate"
    mock_recognizer.nlp.return_value = [
        SimpleNamespace(text="terminate", is_digit=False)
    ]
    mock_recognizer_class.return_value = mock_recognizer

    # 3. Run parser
    parser = OfflineVoiceParser(voice_config)
    command, text = parser.parse_voice_command_from_audio(b"dummy audio")

    # 4. Assertions
    assert text == "Invalid command structure"
    assert command["action"] is None
    assert command["target"] is None
    assert command["id"] is None
    assert "Command failed final validation" in caplog.text
    
@patch('spacy.load')
@patch('vosk.KaldiRecognizer')
@patch('vosk.Model')
@patch('pyaudio.PyAudio')
def test_offline_voice_parser_cache_thread_safety(mock_pyaudio, mock_model, mock_kaldi, mock_spacy, voice_config, caplog):
    """Test thread-safe caching of parsed commands."""

    # Mock recognizer
    mock_recognizer = Mock()
    mock_recognizer.AcceptWaveform.return_value = True
    mock_recognizer.FinalResult.return_value = '{"text": "deploy nginx"}'
    mock_kaldi.return_value = mock_recognizer

    # Mock NLP doc
    tokens = [
        SimpleNamespace(text="deploy", is_digit=False),
        SimpleNamespace(text="nginx", is_digit=False),
        SimpleNamespace(text="service", is_digit=False)
    ]
    mock_spacy.return_value = lambda text: tokens

    # Import parser and create instance
    parser = OfflineVoiceParser(voice_config)

    # Patch recognizer to avoid actual audio decoding
    parser.audio_processor.process_audio_from_bytes = Mock(return_value=np.ones(16000, dtype=np.float32))
    parser.recognizer.recognize_speech = Mock(return_value="deploy nginx")

    def call_parser():
        command, text = parser.parse_voice_command_from_audio(b"dummy audio")
        assert command["action"] == "deploy"
        assert command["image"] == "nginx"
        assert command["target"] == "deployment"
        assert text == "deploy nginx"

    # Start threads to simulate concurrent access
    threads = [threading.Thread(target=call_parser) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert "Successfully parsed voice command: deploy nginx" in caplog.text
    
class FakeTimeoutContext(AbstractContextManager):
    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self  # Let parser enter the block

    def __exit__(self, exc_type, exc_value, traceback):
        raise TimeoutError("Operation timed out")
    
@freeze_time("2025-07-03T10:34:00Z")
@patch('pyaudio.PyAudio')
@patch('vosk.Model')
@patch('vosk.KaldiRecognizer')
@patch('spacy.load')
def test_offline_voice_parser_timeout(mock_spacy, mock_kaldi, mock_model, mock_pyaudio, voice_config, valid_audio, caplog):
    """Test graceful handling of timeout exception."""

    # Patch recognizer to raise TimeoutError inside timeout context
    with patch('src.voice.voice_parser.LocalAudioProcessor.process_audio_from_bytes', return_value=np.ones(16000, dtype=np.float32)), \
         patch('src.voice.voice_parser.OfflineVoiceRecognizer.recognize_speech', side_effect=TimeoutError("Operation timed out")):

        parser = OfflineVoiceParser(voice_config)
        command, text = parser.parse_voice_command_from_audio(valid_audio, timeout=1)

        # ✅ Expected output from catch block
        assert command["action"] is None
        assert text == "Error: Operation timed out"
        assert "Timeout parsing voice command: Operation timed out" in caplog.text
        
@patch('pyaudio.PyAudio')
@patch('vosk.Model')
@patch('vosk.KaldiRecognizer')
@patch('spacy.load')
def test_offline_voice_parser_resource_cleanup(mock_spacy, mock_kaldi, mock_model, mock_pyaudio, voice_config, caplog):
    """Test proper resource cleanup."""
    # Clear Singleton instances to ensure a fresh LocalAudioProcessor
    LocalAudioProcessor._instances.clear()
    
    parser = OfflineVoiceParser(voice_config)
    with parser:
        assert parser.audio_processor._initialized
    assert not parser.audio_processor._initialized
    mock_pyaudio.return_value.terminate.assert_called_once()
    assert "Failed to terminate audio" not in caplog.text
    