import os
import re
import logging
import time
import threading
import hashlib
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import pyaudio
import webrtcvad
import spacy
from collections import deque
import librosa
import soundfile as sf
from cachetools import TTLCache
from vosk import Model, KaldiRecognizer
import unicodedata
import io
import wave

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class VoiceCommandConfig:
    """Configuration for voice command parsing"""
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    format: int = pyaudio.paInt16
    max_audio_duration: int = 30
    silence_threshold: float = 0.01
    min_speech_duration: float = 0.5
    max_silence_duration: float = 2.0
    vad_mode: int = 3
    vad_frame_duration: int = 30
    max_text_length: int = 500
    recognition_timeout: int = 10
    max_cpu: int = 32
    min_cpu: int = 1 
    max_memory_gb: int = 32
    max_storage_gb: int = 1000
    max_ports: int = 10
    cache_ttl: int = 3600
    max_audio_size_mb: int = 10
    allowed_images: List[str] = field(default_factory=lambda: [
        'nginx', 'ubuntu', 'python', 'node', 'redis', 'postgres', 'mysql', 'mongo'
    ])
    allowed_actions: List[str] = field(default_factory=lambda: [
        'deploy', 'start', 'create', 'status', 'check', 'get', 'stop', 'terminate', 'delete'
    ])
    allowed_targets: List[str] = field(default_factory=lambda: [
        'deployment', 'container', 'service', 'app', 'application'
    ])

class VoiceParsingError(Exception):
    """Custom exception for voice parsing errors"""
    pass

class SecurityViolationError(Exception):
    """Raised when security constraints are violated"""
    pass

class Singleton:
    """Singleton pattern implementation"""
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__new__(cls)
            return cls._instances[cls]

class LocalAudioProcessor(Singleton):
    """Local audio processing with VAD and noise reduction"""
    
    def __init__(self, config: VoiceCommandConfig):
        if not hasattr(self, '_initialized'):
            self.config = config
            self.vad = webrtcvad.Vad(config.vad_mode)
            self.audio = pyaudio.PyAudio()
            self._lock = threading.Lock()
            self._initialized = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Explicit cleanup of audio resources"""
        with self._lock:
            try:
                self.audio.terminate()
                self._initialized = False
            except Exception as e:
                logger.error(f"Failed to terminate audio: {e}")

    def _is_speech(self, audio_data: bytes) -> bool:
        """Use VAD to detect speech in audio frame"""
        try:
            if not isinstance(audio_data, bytes):
                raise ValueError("Audio data must be bytes")
            return self.vad.is_speech(audio_data, self.config.sample_rate)
        except (ValueError, webrtcvad.Error) as e:
            logger.warning(f"VAD error: {e}")
            return False

    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio with noise reduction and normalization"""
        try:
            if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
                raise ValueError("Invalid audio data")
            
            audio_data = audio_data.astype(np.float32)
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
            audio_data[np.abs(audio_data) < self.config.silence_threshold] = 0

            from scipy.signal import butter, filtfilt
            nyquist = self.config.sample_rate / 2
            low_freq = 80
            b, a = butter(4, low_freq / nyquist, btype='high')
            audio_data = filtfilt(b, a, audio_data)
            return audio_data
        except (ValueError, ImportError, RuntimeError) as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            return audio_data

    def process_audio_from_bytes(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Process audio data from bytes (e.g., from API upload)"""
        with self._lock:
            try:
                # Validate audio size
                if len(audio_bytes) > self.config.max_audio_size_mb * 1024 * 1024:
                    raise ValueError(f"Audio size exceeds {self.config.max_audio_size_mb}MB limit")

                # Read audio from bytes
                with io.BytesIO(audio_bytes) as f:
                    with wave.open(f, 'rb') as wav:
                        if wav.getnchannels() != self.config.channels:
                            raise ValueError("Audio must be mono")
                        if wav.getframerate() != self.config.sample_rate:
                            raise ValueError(f"Audio must have {self.config.sample_rate}Hz sample rate")
                        if wav.getsampwidth() != 2:  # 16-bit
                            raise ValueError("Audio must be 16-bit PCM")
                        audio_data = wav.readframes(wav.getnframes())
                
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_np = self._preprocess_audio(audio_np)
                logger.info(f"Processed {len(audio_np)/self.config.sample_rate:.2f} seconds of audio")
                return audio_np

            except (ValueError, wave.Error, Exception) as e:
                logger.error(f"Audio processing failed: {e}")
                return None

class OfflineVoiceRecognizer:
    """Offline voice recognition using Vosk"""
    
    def __init__(self, config: VoiceCommandConfig):
        self.config = config
        self.nlp = self._load_nlp_model()
        try:
            self.model = Model("model")
            self.recognizer = KaldiRecognizer(self.model, config.sample_rate)
        except Exception as e:
            logger.error(f"Failed to initialize Vosk: {e}")
            raise VoiceParsingError("ASR model not available")
    
    def _load_nlp_model(self):
        """Load spaCy NLP model"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError as e:
            logger.error(f"spaCy model not found: {e}")
            raise VoiceParsingError("NLP model not available")

    def recognize_speech(self, audio_data: np.ndarray) -> str:
        """Recognize speech from audio data using Vosk"""
        if audio_data is None or len(audio_data) == 0:
            return ""
        
        try:
            audio_int16 = (audio_data * 32768).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            self.recognizer.AcceptWaveform(audio_bytes)
            result = json.loads(self.recognizer.FinalResult())
            text = result.get('text', '').strip().lower()
            
            if not text:
                return ""
            
            logger.info(f"Recognized text: {text}")
            return text
            
        except (ValueError, json.JSONDecodeError, Exception) as e:
            logger.error(f"Speech recognition failed: {e}")
            return ""

class VoiceCommandValidator:
    """Validates and sanitizes voice command parameters"""
    
    def __init__(self, config: VoiceCommandConfig):
        self.config = config
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup regex patterns for validation"""
        self.patterns = {
            'deployment_id': re.compile(r'^[a-zA-Z0-9_-]{1,50}$'),
            'image_name': re.compile(r'^[a-zA-Z0-9._/-]{1,100}$'),
            'memory_size': re.compile(r'^\d+(\.\d+)?(Mi|Gi|MB|GB)$', re.IGNORECASE),
            'storage_size': re.compile(r'^\d+(\.\d+)?(Mi|Gi|MB|GB)$', re.IGNORECASE),
            'port_number': re.compile(r'^([1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5])$'),
            'cpu_value': re.compile(r'^\d+$'),
            'unicode_safe': re.compile(r'^[\x20-\x7E]*$')
        }

    def validate_deployment_id(self, deployment_id: str) -> bool:
        """Validate deployment ID format"""
        if not deployment_id or len(deployment_id) > 50:
            return False
        return bool(self.patterns['deployment_id'].match(deployment_id))

    def validate_image_name(self, image: str) -> bool:
        """Validate container image name"""
        if not image or image not in self.config.allowed_images:
            return False
        return bool(self.patterns['image_name'].match(image))

    def validate_cpu(self, cpu: int) -> bool:
        """Validate CPU allocation"""
        return isinstance(cpu, int) and self.config.min_cpu <= cpu <= self.config.max_cpu

    def validate_memory(self, memory: str) -> bool:
        """Validate memory allocation"""
        if not self.patterns['memory_size'].match(memory):
            return False
        try:
            value_str = re.findall(r'\d+(?:\.\d+)?', memory)[0]
            value = float(value_str)
            unit = memory[-2:].upper()
            mb_value = value * 1024 if unit in ['GI', 'GB'] else value
            return mb_value <= (self.config.max_memory_gb * 1024)
        except (IndexError, ValueError) as e:
            logger.warning(f"Invalid memory value format: {memory} ({e})")
            return False

    def validate_storage(self, storage: str) -> bool:
        """Validate storage allocation"""
        if not self.patterns['storage_size'].match(storage):
            return False
        try:
            value_str = re.findall(r'\d+(?:\.\d+)?', storage)[0]
            value = float(value_str)
            unit = storage[-2:].upper()
            gb_value = value / 1024 if unit in ['MI', 'MB'] else value
            return gb_value <= self.config.max_storage_gb
        except (IndexError, ValueError) as e:
            logger.warning(f"Invalid storage value format: {storage} ({e})")
            return False

    def validate_ports(self, ports: List[str]) -> bool:
        """Validate port numbers"""
        if not ports or len(ports) > self.config.max_ports:
            return False
        for port in ports:
            if not self.patterns['port_number'].match(port):
                return False
            port_num = int(port)
            if port_num < 1 or port_num > 65535:
                return False
        return True

    def sanitize_text(self, text: str) -> str:
        """Sanitize input text"""
        if not text:
            return ""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[^\x20-\x7E]', '', text)
        return text[:self.config.max_text_length].strip()

class OfflineVoiceParser:
    """Offline voice command parser with local speech recognition"""
    
    def __init__(self, config: Optional[VoiceCommandConfig] = None):
        self.config = config or VoiceCommandConfig()
        self.validator = VoiceCommandValidator(self.config)
        self.audio_processor = LocalAudioProcessor(self.config)
        self.recognizer = OfflineVoiceRecognizer(self.config)
        self.nlp = self.recognizer.nlp
        self._command_cache = TTLCache(maxsize=100, ttl=self.config.cache_ttl)
        self._lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.audio_processor.close()

    def _generate_command_hash(self, text: str) -> str:
        """Generate hash for command caching"""
        return hashlib.sha256(text.encode()).hexdigest()

    @contextmanager
    def _timeout_context(self, timeout: int):
        """Context manager for operation timeout"""
        event = threading.Event()
        def timeout_handler():
            event.set()
        timer = threading.Timer(timeout, timeout_handler)
        timer.start()
        try:
            yield event
        finally:
            timer.cancel()

    def _extract_action(self, doc) -> Optional[str]:
        """Extract action from NLP document"""
        for token in doc:
            token_text = token.text.lower()
            if token_text in self.config.allowed_actions:
                if token_text in ["deploy", "start", "create"]:
                    return "deploy"
                elif token_text in ["status", "check", "get"]:
                    return "status"
                elif token_text in ["stop", "terminate", "delete"]:
                    return "terminate"
        return None

    def _extract_target(self, doc) -> Optional[str]:
        """Extract target from NLP document"""
        for token in doc:
            if token.text.lower() in self.config.allowed_targets:
                return "deployment"
        return None

    def _extract_deployment_id(self, doc) -> Optional[str]:
        """Extract deployment ID with validation"""
        for i, token in enumerate(doc):
            if token.text.lower() in ["id", "number"] and i + 1 < len(doc):
                candidate_id = doc[i + 1].text
                if self.validator.validate_deployment_id(candidate_id):
                    return candidate_id
                logger.warning(f"Invalid deployment ID: {candidate_id}")
        return None

    def _extract_image(self, doc) -> str:
        """Extract container image with validation"""
        for i, token in enumerate(doc):
            token_text = token.text.lower()
            if token_text in ["app", "application", "image", "container"] and i + 1 < len(doc):
                image = doc[i + 1].text.lower()
                if self.validator.validate_image_name(image):
                    return image
            if token_text in self.config.allowed_images:
                return token_text
        return "nginx"

    def _extract_cpu(self, doc) -> int:
        """Extract CPU allocation with validation"""
        for i, token in enumerate(doc):
            if token.text.lower() in ["cpu", "processor", "cores"]:
                # Check token after "cpu"
                if i + 1 < len(doc):
                    next_token = doc[i + 1]
                    try:
                        if hasattr(next_token, 'is_digit') and next_token.is_digit or next_token.text.isdigit():
                            cpu_value = int(next_token.text)
                            if self.validator.validate_cpu(cpu_value):
                                logger.debug(f"Extracted CPU value after 'cpu': {cpu_value}")
                                return cpu_value
                            logger.warning(f"CPU value out of range: {cpu_value} (min: {self.config.min_cpu}, max: {self.config.max_cpu})")
                        else:
                            logger.debug(f"Next token after 'cpu' is not numeric: {next_token.text}")
                    except ValueError:
                        logger.warning(f"Invalid CPU value after 'cpu': {next_token.text}")
                # Check token before "cpu"
                if i > 0:
                    prev_token = doc[i - 1]
                    try:
                        if hasattr(prev_token, 'is_digit') and prev_token.is_digit or prev_token.text.isdigit():
                            cpu_value = int(prev_token.text)
                            if self.validator.validate_cpu(cpu_value):
                                logger.debug(f"Extracted CPU value before 'cpu': {cpu_value}")
                                return cpu_value
                            logger.warning(f"CPU value out of range: {cpu_value} (min: {self.config.min_cpu}, max: {self.config.max_cpu})")
                        else:
                            logger.debug(f"Previous token before 'cpu' is not numeric: {prev_token.text}")
                    except ValueError:
                        logger.warning(f"Invalid CPU value before 'cpu': {prev_token.text}")
        logger.debug("No valid CPU value found, returning default: 1")
        return 1

    def _extract_memory(self, doc) -> str:
        """Extract memory allocation with validation"""
        for i, token in enumerate(doc):
            token_text = token.text.lower()
            if token_text in ["memory", "ram", "gigabyte", "gb", "mb"] and i + 1 < len(doc):
                memory_str = doc[i + 1].text
                if memory_str.isdigit():
                    memory_str = f"{memory_str}Mi"
                if self.validator.validate_memory(memory_str):
                    return memory_str
                logger.warning(f"Invalid memory value: {memory_str}")
            if token_text.isdigit() and i + 1 < len(doc):
                unit = doc[i + 1].text.lower()
                if unit in ["gigabyte", "gb", "gi"]:
                    memory_str = f"{token_text}Gi"
                elif unit in ["megabyte", "mb", "mi"]:
                    memory_str = f"{token_text}Mi"
                else:
                    continue
                if self.validator.validate_memory(memory_str):
                    return memory_str
        return "512Mi"

    def _extract_storage(self, doc) -> str:
        """Extract storage allocation with validation"""
        for i, token in enumerate(doc):
            token_text = token.text.lower()
            if token_text in ["storage", "disk", "volume"] and i + 1 < len(doc):
                storage_str = doc[i + 1].text
                if storage_str.isdigit():
                    storage_str = f"{storage_str}Mi"
                if self.validator.validate_storage(storage_str):
                    return storage_str
                logger.warning(f"Invalid storage value: {storage_str}")
        return "512Mi"

    def _extract_ports(self, doc) -> List[str]:
        """Extract port numbers with validation"""
        ports = []
        for i, token in enumerate(doc):
            if token.text.lower() in ["port", "ports"] and i + 1 < len(doc):
                if doc[i + 1].is_digit:
                    port = doc[i + 1].text
                    if self.validator.validate_ports([port]):
                        ports.append(port)
                    else:
                        logger.warning(f"Invalid port number: {port}")
        return ports if ports else ["80"]

    def _create_default_command(self) -> Dict[str, Any]:
        """Create default secure command structure"""
        return {
            "action": None,
            "target": None,
            "id": None,
            "image": "nginx",
            "cpu": 1,
            "memory": "512Mi",
            "storage": "512Mi",
            "ports": ["80"]
        }

    def parse_voice_command_from_audio(self, audio_bytes: bytes, timeout: int = 10) -> Tuple[Dict[str, Any], str]:
        """Parse voice command from audio bytes"""
        with self._timeout_context(timeout) as event:
            try:
                audio_data = self.audio_processor.process_audio_from_bytes(audio_bytes)
                if audio_data is None:
                    return self._create_default_command(), "No valid audio data"
                text = self.recognizer.recognize_speech(audio_data)
                if not text:
                    return self._create_default_command(), "Could not understand audio"
                return self.parse_voice_command_from_text(text)
            except (TimeoutError, threading.ThreadError) as e:
                logger.error(f"Timeout parsing voice command: {e}")
                return self._create_default_command(), f"Error: {e}"

    def parse_voice_command_from_text(self, text: str) -> Tuple[Dict[str, Any], str]:
        """Parse voice command from text input"""
        with self._lock:
            try:
                text = self.validator.sanitize_text(text)
                if not text:
                    return self._create_default_command(), "Empty or invalid input"
                
                text_hash = self._generate_command_hash(text)
                if text_hash in self._command_cache:
                    logger.debug("Using cached command parsing result")
                    return self._command_cache[text_hash], text
                
                doc = self.nlp(text.lower())
                command = self._create_default_command()
                
                command["action"] = self._extract_action(doc)
                command["target"] = self._extract_target(doc)
                command["id"] = self._extract_deployment_id(doc)
                command["image"] = self._extract_image(doc)
                command["cpu"] = self._extract_cpu(doc)
                command["memory"] = self._extract_memory(doc)
                command["storage"] = self._extract_storage(doc)
                command["ports"] = self._extract_ports(doc)
                
                if not self._validate_command(command):
                    logger.warning("Command failed final validation")
                    return self._create_default_command(), "Invalid command structure"
                
                self._command_cache[text_hash] = command
                logger.info(f"Successfully parsed voice command: {text}")
                return command, text
                
            except Exception as e:
                logger.error(f"Error parsing voice command from text: {e}")
                return self._create_default_command(), f"Error: {e}"

    def _validate_command(self, command: Dict[str, Any]) -> bool:
        """Final validation of complete command"""
        try:
            if not self.validator.validate_image_name(command["image"]):
                return False
            if not self.validator.validate_cpu(command["cpu"]):
                return False
            if not self.validator.validate_memory(command["memory"]):
                return False
            if not self.validator.validate_storage(command["storage"]):
                return False
            if not self.validator.validate_ports(command["ports"]):
                return False
            if command["action"] == "terminate" and not command["id"]:
                logger.warning("Terminate action requires deployment ID")
                return False
            if command["action"] in ["deploy", "start", "create"] and not command["target"]:
                logger.warning("Deploy action requires target")
                return False
            return True
        except Exception as e:
            logger.error(f"Error in command validation: {e}")
            return False

    def clear_cache(self):
        """Clear command cache"""
        with self._lock:
            self._command_cache.clear()
            logger.info("Command cache cleared")

def create_offline_voice_parser(config: Optional[VoiceCommandConfig] = None) -> OfflineVoiceParser:
    """Create an offline voice parser instance"""
    return OfflineVoiceParser(config)

def parse_voice_command(audio_bytes: bytes) -> Tuple[Dict[str, Any], str]:
    """Parse voice command from audio bytes (backward compatibility)"""
    with create_offline_voice_parser() as parser:
        return parser.parse_voice_command_from_audio(audio_bytes)