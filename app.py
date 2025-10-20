#!/usr/bin/env python3
"""
Universal Live Translator Desktop — Professional Edition v3.0
-------------------------------------------------------------
✨ Advanced Features:
- 🎙️ Continuous listening (never stops!)
- 📺 Fully resizable overlay (drag corners/edges)
- 🚀 GPU acceleration (CUDA/MPS)
- ⚡ Performance optimizations (async pipeline)
- 💎 Professional UX (glassmorphic design)
"""
import os, sys, threading, queue, logging, sqlite3, tempfile, json, time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
from collections import deque
import hashlib
import numpy as np
import warnings
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import subprocess

REQUIRED_PACKAGES = [
    "PyQt6", "pyttsx3", "gtts", "argostranslate", "SpeechRecognition",
    "sounddevice", "vosk", "pydub", "langdetect", "deep-translator",
    "simpleaudio", "numpy", "requests", "openai-whisper", "torch", "pyaudio", "scipy"
]

def ensure_dependencies():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg.replace('-', '_').split('[')[0])
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"\n{'='*60}\nMissing Dependencies\n{'='*60}")
        print("The following packages need to be installed:")
        for pkg in missing:
            print(f"  • {pkg}")
        print(f"\nTotal size: ~500MB (includes PyTorch and Whisper)\n{'='*60}")
        
        response = input("\nInstall missing packages? (y/n): ").strip().lower()
        if response == 'y':
            for pkg in missing:
                print(f"\n📦 Installing {pkg}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
            print("\n✅ All dependencies installed!\n")
        else:
            print("\n❌ Cannot proceed without packages. Exiting...")
            sys.exit(1)

ensure_dependencies()

import pyttsx3
from gtts import gTTS
import speech_recognition as sr
import sounddevice as sd
import pyaudio
from vosk import Model, KaldiRecognizer
from langdetect import detect, LangDetectException, detect_langs
from deep_translator import GoogleTranslator
from deep_translator.constants import GOOGLE_LANGUAGES_TO_CODES
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from pydub import AudioSegment
import simpleaudio as sa
import argostranslate.translate
import argostranslate.package
import requests
import whisper
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translator.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("Translator")

BASE_DIR = Path.home() / ".universal_translator"
BASE_DIR.mkdir(exist_ok=True, parents=True)

DB_FILE = BASE_DIR / "translator_history.db"
AUDIO_DIR = BASE_DIR / "audio_cache"
VOSK_MODELS_DIR = BASE_DIR / "vosk_models"
CONFIG_FILE = BASE_DIR / "translator_config.json"

AUDIO_DIR.mkdir(exist_ok=True)
VOSK_MODELS_DIR.mkdir(exist_ok=True)

class RecognitionEngine(Enum):
    GOOGLE = "google"
    VOSK = "vosk"
    WHISPER = "whisper"

VOSK_MODELS = {
    "en": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "fr": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "es": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
    "de": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

DEFAULT_CONFIG = {
    "theme": "dark", "overlay_position": [100, 100, 800, 150], "overlay_visible": True,
    "font_size": 20, "text_color": "#FFFFFF", "bg_color": "rgba(20,20,30,0.85)",
    "max_words": 100, "auto_speak": True, "tts_rate": 150, "volume": 0.8,
    "cache_translations": True, "cache_expiry_days": 7, "source_language": "auto",
    "target_language": "fr", "show_confidence": True, "recognition_engine": "google",
    "whisper_model": "base", "audio_device_input": "default", "use_gpu": True,
    "animation_duration": 200, "live_captions_mode": True, "subtitle_lines": 3,
    "subtitle_update_delay": 10  # milliseconds between word updates (lower = faster)
}

@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_input: bool
    is_output: bool
    host_api: str

class GPUManager:
    """Manages GPU detection and device selection"""
    def __init__(self):
        # Enhanced CUDA detection
        self.has_cuda = torch.cuda.is_available()
        if not self.has_cuda and torch.version.cuda is not None:
            # Try to initialize CUDA even if not initially available
            try:
                torch.cuda.init()
                self.has_cuda = torch.cuda.is_available()
            except:
                pass
        
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        self.device = self._detect_device()
        self.device_name = self._get_device_name()
        
        # Log detailed GPU info
        if self.has_cuda:
            log.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
            log.info(f"CUDA version: {torch.version.cuda}")
            log.info(f"CUDA device count: {torch.cuda.device_count()}")
        else:
            log.warning("CUDA not detected. GPU acceleration disabled.")
            if torch.version.cuda:
                log.warning(f"PyTorch built with CUDA {torch.version.cuda} but CUDA runtime not available")
    
    def _detect_device(self):
        if self.has_cuda:
            return "cuda"
        elif self.has_mps:
            return "mps"
        return "cpu"
    
    def _get_device_name(self):
        if self.has_cuda:
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        elif self.has_mps:
            return "Apple Silicon (MPS)"
        return "CPU"
    
    def get_device(self, use_gpu=True):
        if use_gpu and (self.has_cuda or self.has_mps):
            return self.device
        return "cpu"
    
    def is_gpu_available(self):
        return self.has_cuda or self.has_mps

gpu_manager = GPUManager()

class AudioDeviceManager:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.devices = self._detect_devices()
        self.input_devices = [d for d in self.devices if d.is_input]
    
    def _detect_devices(self) -> List[AudioDevice]:
        devices = []
        try:
            for i in range(self.pa.get_device_count()):
                info = self.pa.get_device_info_by_index(i)
                device = AudioDevice(
                    index=i, name=info['name'],
                    max_input_channels=info['maxInputChannels'],
                    max_output_channels=info['maxOutputChannels'],
                    default_samplerate=info['defaultSampleRate'],
                    is_input=info['maxInputChannels'] > 0,
                    is_output=info['maxOutputChannels'] > 0,
                    host_api=self.pa.get_host_api_info_by_index(info['hostApi'])['name']
                )
                devices.append(device)
        except Exception as e:
            log.error(f"Error detecting devices: {e}")
        return devices
    
    def get_default_input_device(self) -> Optional[AudioDevice]:
        try:
            default_info = self.pa.get_default_input_device_info()
            for device in self.input_devices:
                if device.index == default_info['index']:
                    return device
        except:
            pass
        return self.input_devices[0] if self.input_devices else None
    
    def get_loopback_device(self) -> Optional[AudioDevice]:
        """Get loopback device with validation"""
        keywords = ['stereo mix', 'loopback', 'wave out', 'what u hear', 'wave-out', 'waveout']
        candidates = []
        
        # Find all potential loopback devices
        for device in self.input_devices:
            device_name_lower = device.name.lower()
            if any(k in device_name_lower for k in keywords):
                candidates.append(device)
        
        # Try to find a working device by testing each candidate
        for device in candidates:
            try:
                # Quick test with sounddevice instead of pyaudio
                import sounddevice as sd
                test_stream = sd.InputStream(
                    channels=1,
                    samplerate=int(device.default_samplerate),
                    device=device.index,
                    blocksize=1024
                )
                test_stream.close()
                log.info(f"Found working loopback device: {device.name}")
                return device
            except Exception as e:
                log.warning(f"Loopback device {device.name} failed test: {e}")
                continue
        
        # If no keywords match, try the default input device if it's not a microphone
        try:
            default = self.get_default_input_device()
            if default and 'mic' not in default.name.lower():
                log.info(f"Trying default input as loopback: {default.name}")
                return default
        except Exception as e:
            log.warning(f"Default device check failed: {e}")
        
        return None
    
    def test_device(self, device: AudioDevice, duration: float = 1.0) -> bool:
        try:
            stream = self.pa.open(format=pyaudio.paInt16, channels=1,
                rate=int(device.default_samplerate), input=True,
                input_device_index=device.index, frames_per_buffer=1024)
            stream.read(int(device.default_samplerate * duration))
            stream.stop_stream()
            stream.close()
            return True
        except:
            return False
    
    def cleanup(self):
        self.pa.terminate()

audio_device_manager = AudioDeviceManager()

class ConfigManager:
    def __init__(self):
        self.config = self.load_config()
        self.lock = threading.Lock()
    
    def load_config(self) -> dict:
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return {**DEFAULT_CONFIG, **json.load(f)}
            except:
                pass
        return DEFAULT_CONFIG.copy()
    
    def save_config(self):
        with self.lock:
            try:
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(self.config, f, indent=2)
            except Exception as e:
                log.error(f"Save config failed: {e}")
    
    def get(self, key: str, default=None):
        with self.lock:
            return self.config.get(key, default)
    
    def set(self, key: str, value):
        with self.lock:
            self.config[key] = value
        self.save_config()

config = ConfigManager()

class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self.init_tables()
    
    def _get_connection(self):
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=True)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def init_tables(self):
        conn = self._get_connection()
        conn.execute("""CREATE TABLE IF NOT EXISTS translations (
            id INTEGER PRIMARY KEY, source_text TEXT, translated_text TEXT,
            source_lang TEXT, target_lang TEXT, mode TEXT, engine TEXT,
            confidence REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, duration_ms INTEGER)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS cache (
            id INTEGER PRIMARY KEY, hash TEXT UNIQUE, source_text TEXT, translated_text TEXT,
            source_lang TEXT, target_lang TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 0, last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        conn.commit()
    
    def add_translation(self, source, translated, src_lang, tgt_lang, mode, engine="google", confidence=1.0, duration_ms=0):
        conn = self._get_connection()
        conn.execute("INSERT INTO translations (source_text, translated_text, source_lang, target_lang, mode, engine, confidence, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (source, translated, src_lang, tgt_lang, mode, engine, confidence, duration_ms))
        conn.commit()
    
    def get_cached_translation(self, text, src, tgt):
        key = hashlib.md5(f"{text}:{src}:{tgt}".encode()).hexdigest()
        conn = self._get_connection()
        result = conn.execute("SELECT translated_text FROM cache WHERE hash = ?", (key,)).fetchone()
        return result[0] if result else None
    
    def cache_translation(self, text, translated, src, tgt):
        key = hashlib.md5(f"{text}:{src}:{tgt}".encode()).hexdigest()
        conn = self._get_connection()
        try:
            conn.execute("INSERT OR REPLACE INTO cache (hash, source_text, translated_text, source_lang, target_lang, last_accessed) VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (key, text, translated, src, tgt))
            conn.commit()
        except:
            pass
    
    def get_history(self, limit=50, search=""):
        conn = self._get_connection()
        if search:
            results = conn.execute("SELECT source_text, translated_text, source_lang, target_lang, mode, engine, confidence, timestamp FROM translations WHERE source_text LIKE ? OR translated_text LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{search}%", f"%{search}%", limit)).fetchall()
        else:
            results = conn.execute("SELECT source_text, translated_text, source_lang, target_lang, mode, engine, confidence, timestamp FROM translations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in results]
    
    def export_history(self, filepath):
        conn = self._get_connection()
        results = conn.execute("SELECT * FROM translations ORDER BY timestamp DESC").fetchall()
        with open(filepath, 'w', encoding='utf-8') as f:
            for row in results:
                r = dict(row)
                f.write(f"[{r['timestamp']}] {r['source_lang']}→{r['target_lang']} ({r['mode']}/{r['engine']})\n")
                f.write(f"Source: {r['source_text']}\nTranslation: {r['translated_text']}\n{'-'*80}\n\n")
    
    def clear_history(self):
        conn = self._get_connection()
        conn.execute("DELETE FROM translations")
        conn.commit()

db = DatabaseManager(DB_FILE)

def is_online():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

class WhisperModelManager:
    def __init__(self):
        self.models = {}
        self.device = "cpu"
        self.use_gpu = config.get("use_gpu", True)
        # Initialize device on startup - force GPU if available
        self.set_device(self.use_gpu)
        log.info(f"WhisperModelManager initialized with device: {self.device} (use_gpu={self.use_gpu})")
    
    def set_device(self, use_gpu=True):
        self.use_gpu = use_gpu
        old_device = self.device
        self.device = gpu_manager.get_device(use_gpu)
        log.info(f"WhisperModelManager device set to: {self.device} (requested use_gpu={use_gpu})")
        log.info(f"GPU available: {gpu_manager.is_gpu_available()}, CUDA: {gpu_manager.has_cuda}, MPS: {gpu_manager.has_mps}")
        
        # Reload models if device changed
        if old_device != self.device and self.models:
            log.info(f"Device changed from {old_device} to {self.device}, reloading models...")
            old_models = list(self.models.keys())
            self.models.clear()
            for model_size in old_models:
                self.load_model(model_size)
    
    def load_model(self, model_size="base"):
        if model_size in self.models:
            return True
        try:
            log.info(f"Loading Whisper {model_size} on {self.device}...")
            self.models[model_size] = whisper.load_model(model_size, device=self.device)
            log.info(f"Whisper {model_size} loaded successfully")
            return True
        except Exception as e:
            log.error(f"Whisper load failed: {e}")
            return False
    
    def transcribe(self, audio_data, model_size="base", language=None):
        if model_size not in self.models:
            if not self.load_model(model_size):
                return "", 0.0
        try:
            # Validate audio data
            if audio_data is None or len(audio_data) == 0:
                log.warning("Empty audio data received")
                return "", 0.0
            
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                if np.abs(audio_data).max() > 1.0:
                    audio_data = audio_data / 32768.0
            
            # Ensure minimum audio length (Whisper needs at least some samples)
            if len(audio_data) < 400:  # ~0.025s at 16kHz
                log.warning(f"Audio too short: {len(audio_data)} samples")
                return "", 0.0
            
            fp16 = self.device == "cuda"
            result = self.models[model_size].transcribe(
                audio_data, 
                language=language, 
                task="transcribe", 
                fp16=fp16,
                condition_on_previous_text=False  # Avoid issues with empty audio
            )
            text = result.get("text", "").strip()
            segments = result.get("segments", [])
            confidence = 1.0 - (sum(s.get("no_speech_prob", 0) for s in segments) / len(segments) if segments else 0)
            return text, confidence
        except Exception as e:
            log.error(f"Whisper transcribe failed: {e}")
            import traceback
            log.error(traceback.format_exc())
            return "", 0.0

whisper_manager = WhisperModelManager()

class VoskModelManager:
    def __init__(self):
        self.models = {}
        self.load_existing_models()
    
    def load_existing_models(self):
        for lang_code in VOSK_MODELS.keys():
            model_dir = VOSK_MODELS_DIR / f"vosk-model-small-{lang_code}"
            if model_dir.exists():
                try:
                    self.models[lang_code] = Model(str(model_dir))
                except:
                    pass
    
    def get_model(self, lang_code):
        return self.models.get(lang_code)
    
    def download_model(self, lang_code, progress_callback=None):
        if lang_code not in VOSK_MODELS:
            return False
        model_dir = VOSK_MODELS_DIR / f"vosk-model-small-{lang_code}"
        if model_dir.exists():
            return True
        try:
            url = VOSK_MODELS[lang_code]
            zip_path = VOSK_MODELS_DIR / f"model-{lang_code}.zip"
            response = requests.get(url, stream=True)
            total = int(response.headers.get('content-length', 0))
            downloaded = 0
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total > 0:
                            progress_callback(downloaded / total)
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(VOSK_MODELS_DIR)
            zip_path.unlink()
            for item in VOSK_MODELS_DIR.iterdir():
                if item.is_dir() and lang_code in item.name.lower():
                    if item.name != f"vosk-model-small-{lang_code}":
                        item.rename(model_dir)
                    break
            self.models[lang_code] = Model(str(model_dir))
            return True
        except Exception as e:
            log.error(f"Download failed: {e}")
            return False
    
    def has_model(self, lang_code):
        return lang_code in self.models

vosk_manager = VoskModelManager()

def setup_argos():
    try:
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        installed = {l.code for l in argostranslate.translate.get_installed_languages()}
        for from_l in ["en", "fr", "es"]:
            for to_l in ["en", "fr", "es"]:
                if from_l != to_l and from_l not in installed:
                    pkg = next((p for p in available if p.from_code==from_l and p.to_code==to_l), None)
                    if pkg:
                        argostranslate.package.install_from_path(pkg.download())
    except:
        pass

threading.Thread(target=setup_argos, daemon=True).start()

class TranslationEngine:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="TranslateWorker")
    
    def translate(self, text, src, tgt):
        if not text.strip():
            return "", 0, 1.0
        if config.get("cache_translations"):
            cached = db.get_cached_translation(text, src, tgt)
            if cached:
                return cached, 0, 1.0
        start = time.time()
        translated = None
        confidence = 1.0
        if is_online():
            try:
                translated = GoogleTranslator(source=src, target=tgt).translate(text)
            except:
                pass
        if not translated:
            try:
                translated = argostranslate.translate.translate(text, from_code=src, to_code=tgt)
                confidence = 0.85
            except:
                translated = text
                confidence = 0.0
        duration = int((time.time() - start) * 1000)
        if translated and config.get("cache_translations"):
            db.cache_translation(text, translated, src, tgt)
        return translated, duration, confidence
    
    def translate_async(self, text, src, tgt, callback):
        """Async translation for non-blocking processing"""
        def _translate():
            result = self.translate(text, src, tgt)
            callback(result)
        self.executor.submit(_translate)

translator = TranslationEngine()

class TTSManager:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.current = None
        self.volume = config.get("volume", 0.8)
        self.temp_files = []
        threading.Thread(target=self._worker, daemon=True).start()
    
    def _worker(self):
        while self.running:
            try:
                text, lang = self.queue.get(timeout=0.5)
                self._speak(text, lang)
            except queue.Empty:
                continue
            except Exception as e:
                log.warning(f"TTS error: {e}")
    
    def _speak(self, text, lang):
        try:
            if is_online():
                tts = gTTS(text=text, lang=lang, slow=False)
                tmp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=AUDIO_DIR)
                tmp_mp3.close()
                tts.save(tmp_mp3.name)
                tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=AUDIO_DIR)
                tmp_wav.close()
                audio = AudioSegment.from_mp3(tmp_mp3.name)
                audio = audio + (20 * np.log10(self.volume))
                audio.export(tmp_wav.name, format="wav")
                wave_obj = sa.WaveObject.from_wave_file(tmp_wav.name)
                self.current = wave_obj.play()
                self.current.wait_done()
                os.unlink(tmp_mp3.name)
                os.unlink(tmp_wav.name)
            else:
                engine = pyttsx3.init()
                engine.setProperty('rate', config.get("tts_rate", 150))
                engine.say(text)
                engine.runAndWait()
        except Exception as e:
            log.error(f"TTS failed: {e}")
    
    def speak(self, text, lang):
        if text.strip():
            self.queue.put((text, lang))
    
    def stop(self):
        if self.current and self.current.is_playing():
            self.current.stop()
    
    def set_volume(self, volume):
        self.volume = max(0.0, min(1.0, volume))
    
    def shutdown(self):
        self.running = False
        self.stop()

tts_manager = TTSManager()

class ContinuousSpeechRecognitionThread(QThread):
    """Continuous listening - never stops, async processing"""
    phrase_detected = pyqtSignal(str, float)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    performance_update = pyqtSignal(dict)  # Queue sizes, processing time
    
    def __init__(self, source_type="microphone", language="en", engine=RecognitionEngine.GOOGLE, device_index=None):
        super().__init__()
        self.running = True
        self.source_type = source_type
        self.language = language
        self.engine = engine
        self.device_index = device_index
        self.vosk_model = vosk_manager.get_model(language) if engine == RecognitionEngine.VOSK else None
        self.whisper_model_size = config.get("whisper_model", "base")
        
        # Processing queues for async pipeline
        self.recognition_queue = queue.Queue(maxsize=10)  # Audio chunks waiting for recognition
        self.translation_queue = queue.Queue(maxsize=10)  # Texts waiting for translation
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="RecognitionWorker")
        self.executor_shutdown = False
        
        # Performance monitoring
        self.processing_times = deque(maxlen=20)
        self.last_perf_update = time.time()
    
    def run(self):
        try:
            if self.source_type == "microphone":
                self._listen_microphone_continuous()
            else:
                self._listen_system_audio_continuous()
        except Exception as e:
            self.error_occurred.emit(f"Recognition error: {e}")
    
    def _listen_microphone_continuous(self):
        """Continuous microphone listening - never stops"""
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 3000
        recognizer.pause_threshold = 1.2
        
        try:
            kwargs = {}
            if self.device_index is not None:
                kwargs['device_index'] = self.device_index
            
            with sr.Microphone(**kwargs) as source:
                self.status_changed.emit("🎙️ Calibrating...")
                recognizer.adjust_for_ambient_noise(source, 1)
                self.status_changed.emit(f"🎙️ Continuous Listening ({self.engine.value})...")
                
                while self.running:
                    try:
                        # Non-blocking listen - captures audio continuously
                        audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)
                        
                        # Async processing - submit to queue
                        if not self.recognition_queue.full() and not self.executor_shutdown:
                            self.executor.submit(self._process_audio, audio)
                        elif self.executor_shutdown:
                            break
                        else:
                            log.warning("Recognition queue full, dropping audio")
                        
                        # Update performance metrics
                        self._update_performance()
                        
                    except sr.WaitTimeoutError:
                        pass
                    except Exception as e:
                        if self.running:  # Only log if we're supposed to be running
                            log.warning(f"Mic error: {e}")
                        time.sleep(0.5)
        except Exception as e:
            if self.running:
                self.error_occurred.emit(f"Microphone setup failed: {e}")
    
    def _listen_system_audio_continuous(self):
        """Continuous system audio capture with proper threading and validation"""
        loopback = audio_device_manager.get_loopback_device()
        if not loopback:
            error_msg = (
                "❌ No working loopback device found.\n\n"
                "Windows: Enable 'Stereo Mix' in Sound Settings:\n"
                "  1. Right-click speaker icon → Sounds\n"
                "  2. Recording tab → Right-click → Show Disabled Devices\n"
                "  3. Enable 'Stereo Mix' or 'Wave Out Mix'\n\n"
                "macOS: Install BlackHole or Soundflower\n\n"
                "Linux: Use PulseAudio monitor device"
            )
            self.error_occurred.emit(error_msg)
            return
        
        # Validate sample rate
        try:
            samplerate = int(loopback.default_samplerate)
            if samplerate > 48000:
                log.warning(f"Sample rate {samplerate} too high, limiting to 16000")
                samplerate = 16000
            elif samplerate < 8000:
                log.warning(f"Sample rate {samplerate} too low, using 16000")
                samplerate = 16000
            else:
                samplerate = min(samplerate, 16000)
        except Exception as e:
            log.error(f"Sample rate validation failed: {e}")
            samplerate = 16000
        
        audio_buffer = []
        samples_needed = int(samplerate * 5.0)
        buffer_lock = threading.Lock()
        stream_error = [None]  # Use list to allow modification in callback
        
        def callback(indata, frames, time_info, status):
            """Audio callback - runs in separate thread"""
            try:
                if status:
                    log.warning(f"Audio callback status: {status}")
                
                if not self.running:
                    raise sd.CallbackStop()
                
                # Validate input data
                if indata is None or len(indata) == 0:
                    return
                
                with buffer_lock:
                    # Handle mono/stereo input
                    if indata.ndim == 1:
                        audio_buffer.extend(indata.copy())
                    else:
                        audio_buffer.extend(indata[:, 0].copy())
                    
                    if len(audio_buffer) >= samples_needed:
                        chunk = np.array(audio_buffer[:samples_needed])
                        audio_buffer.clear()
                        
                        # Submit to thread pool for processing
                        if not self.recognition_queue.full() and not self.executor_shutdown:
                            try:
                                self.executor.submit(self._process_system_audio, chunk, samplerate)
                            except Exception as e:
                                if self.running:
                                    log.error(f"Failed to submit audio processing: {e}")
            except sd.CallbackStop:
                raise
            except Exception as e:
                stream_error[0] = str(e)
                log.error(f"Audio callback error: {e}")
                raise sd.CallbackStop()
        
        try:
            log.info(f"Opening system audio stream: device={loopback.name}, rate={samplerate}, index={loopback.index}")
            self.status_changed.emit(f"🎧 Capturing system audio ({self.engine.value})...")
            
            # Test device access before starting stream
            try:
                test_stream = sd.InputStream(
                    channels=1,
                    samplerate=samplerate,
                    device=loopback.index,
                    blocksize=1024
                )
                test_stream.close()
                log.info("Device test successful")
            except Exception as test_error:
                error_msg = (
                    f"❌ System audio device test failed:\n\n"
                    f"Device: {loopback.name}\n"
                    f"Error: {test_error}\n\n"
                    f"The device may be in use by another application\n"
                    f"or not properly configured for recording."
                )
                self.error_occurred.emit(error_msg)
                return
            
            # Start actual capture stream
            with sd.InputStream(
                channels=1, 
                samplerate=samplerate, 
                device=loopback.index, 
                callback=callback, 
                blocksize=4096
            ):
                while self.running:
                    if stream_error[0]:
                        raise RuntimeError(stream_error[0])
                    sd.sleep(100)
                    self._update_performance()
                    
        except Exception as e:
            if self.running:
                import traceback
                error_details = traceback.format_exc()
                log.error(f"System audio error:\n{error_details}")
                
                error_msg = (
                    f"❌ System audio failed:\n\n"
                    f"Error: {str(e)}\n\n"
                    f"Device: {loopback.name if loopback else 'Unknown'}\n\n"
                    f"Possible solutions:\n"
                    f"• Close other apps using the audio device\n"
                    f"• Check device is enabled in system settings\n"
                    f"• Try restarting the application\n"
                    f"• Use Microphone mode instead"
                )
                self.error_occurred.emit(error_msg)
    
    def _process_audio(self, audio):
        """Process audio chunk - runs in thread pool"""
        start_time = time.time()
        try:
            text, conf = None, 1.0
            
            if self.engine == RecognitionEngine.WHISPER:
                text, conf = self._recognize_whisper(audio)
            elif self.engine == RecognitionEngine.VOSK and self.vosk_model:
                text, conf = self._recognize_vosk(audio.get_wav_data())
            elif self.engine == RecognitionEngine.GOOGLE:
                if is_online():
                    try:
                        text = sr.Recognizer().recognize_google(audio, language=self.language)
                    except:
                        pass
            
            if text and text.strip():
                self.phrase_detected.emit(text, conf)
            
            # Track processing time
            elapsed = (time.time() - start_time) * 1000
            self.processing_times.append(elapsed)
            
        except Exception as e:
            log.error(f"Process audio error: {e}")
    
    def _process_system_audio(self, audio_data, samplerate):
        """Process system audio chunk"""
        if not self.running or self.executor_shutdown:
            return
        
        start_time = time.time()
        try:
            energy = np.sqrt(np.mean(audio_data ** 2))
            if energy < 0.01:
                return
            
            text, conf = "", 0.0
            if self.engine == RecognitionEngine.WHISPER:
                text, conf = self._transcribe_whisper_array(audio_data, samplerate)
            elif self.engine == RecognitionEngine.VOSK and self.vosk_model:
                audio_int = (audio_data * 32767).astype(np.int16)
                text, conf = self._recognize_vosk(audio_int.tobytes())
            
            if text and text.strip():
                self.phrase_detected.emit(text, conf)
            
            elapsed = (time.time() - start_time) * 1000
            self.processing_times.append(elapsed)
            
        except Exception as e:
            if self.running:
                log.error(f"Process system audio error: {e}")
    
    def _recognize_vosk(self, wav_data):
        if not self.vosk_model:
            return "", 0.0
        try:
            rec = KaldiRecognizer(self.vosk_model, 16000)
            rec.SetWords(True)
            rec.AcceptWaveform(wav_data)
            result = json.loads(rec.FinalResult())
            return result.get("text", ""), result.get("confidence", 0.9)
        except:
            return "", 0.0
    
    def _recognize_whisper(self, audio):
        try:
            audio_np = np.frombuffer(audio.get_wav_data(), dtype=np.int16).astype(np.float32) / 32768.0
            if audio.sample_rate != 16000:
                from scipy import signal
                audio_np = signal.resample(audio_np, int(len(audio_np) * 16000 / audio.sample_rate))
            lang = self.language if self.language != "auto" else None
            return whisper_manager.transcribe(audio_np, self.whisper_model_size, lang)
        except:
            return "", 0.0
    
    def _transcribe_whisper_array(self, audio_data, samplerate):
        try:
            if samplerate != 16000:
                from scipy import signal
                audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / samplerate))
            lang = self.language if self.language != "auto" else None
            return whisper_manager.transcribe(audio_data, self.whisper_model_size, lang)
        except:
            return "", 0.0
    
    def _update_performance(self):
        """Update performance metrics periodically"""
        now = time.time()
        if now - self.last_perf_update > 1.0:  # Update every second
            avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
            perf_data = {
                'recognition_queue': self.recognition_queue.qsize(),
                'translation_queue': self.translation_queue.qsize(),
                'avg_processing_time': avg_time,
                'device': whisper_manager.device if self.engine == RecognitionEngine.WHISPER else 'N/A'
            }
            self.performance_update.emit(perf_data)
            self.last_perf_update = now
    
    def stop(self):
        self.running = False
        self.executor_shutdown = True
        # Shutdown executor gracefully
        self.executor.shutdown(wait=False)

class ResizableOverlay(QWidget):
    """Fully resizable overlay with corner and edge dragging + Google Live Captions-style scrolling"""
    
    CORNER_SIZE = 20
    EDGE_SIZE = 10
    MIN_WIDTH = 135
    MIN_HEIGHT = 57
    
    class ResizeMode(Enum):
        NONE = 0
        MOVE = 1
        TOP_LEFT = 2
        TOP_RIGHT = 3
        BOTTOM_LEFT = 4
        BOTTOM_RIGHT = 5
        TOP = 6
        BOTTOM = 7
        LEFT = 8
        RIGHT = 9
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        
        # Resize state
        self.resize_mode = self.ResizeMode.NONE
        self.drag_start_pos = None
        self.drag_start_geometry = None
        
        # Container with glassmorphic effect
        self.container = QWidget(self)
        self.container.setObjectName("overlayContainer")
        
        # Main content
        # Use a scrolling label for Google Live Captions effect
        self.label_container = QWidget(self.container)
        self.label_container.setMouseTracking(True)
        
        self.label = QLabel("", self.label_container)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        self.label.setMouseTracking(True)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        label_layout = QVBoxLayout(self.label_container)
        label_layout.addStretch()
        label_layout.addWidget(self.label)
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(0)
        
        # Resize indicators (corner dots)
        self.corner_indicators = []
        for _ in range(4):
            indicator = QLabel("", self.container)
            indicator.setFixedSize(8, 8)
            indicator.setStyleSheet("background: rgba(100, 181, 246, 0.6); border-radius: 4px;")
            indicator.hide()
            self.corner_indicators.append(indicator)
        
        layout = QVBoxLayout(self.container)
        layout.addWidget(self.label_container)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(0)
        
        main = QVBoxLayout(self)
        main.addWidget(self.container)
        main.setContentsMargins(0, 0, 0, 0)
        
        # Live captions style: show last N words with smooth scrolling
        self.text_buffer = deque(maxlen=config.get("max_words", 100))
        max_lines = config.get("subtitle_lines", 3)
        self.displayed_lines = deque(maxlen=max_lines)  # Show last N lines like Google Live Captions
        self.current_sentence = []  # Build current sentence
        self.last_update_time = time.time()
        self.live_captions_mode = config.get("live_captions_mode", True)
        self.subtitle_update_delay = config.get("subtitle_update_delay", 10) / 1000.0  # Convert ms to seconds
        
        # Fade animation for smooth text transitions
        self.fade_effect = QGraphicsOpacityEffect(self.label)
        self.label.setGraphicsEffect(self.fade_effect)
        
        self.fade_animation = QPropertyAnimation(self.fade_effect, b"opacity")
        self.fade_animation.setDuration(200)
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        self.apply_style()
        geom = config.get("overlay_position", [100, 100, 800, 180])
        # Ensure geometry is valid
        geom = self._validate_geometry(geom)
        self.setGeometry(*geom)
        self.setVisible(config.get("overlay_visible", True))
    
    def _validate_geometry(self, geom):
        """Validate and fix geometry to avoid Windows constraints errors"""
        x, y, w, h = geom
        
        # Ensure minimum size
        w = max(w, self.MIN_WIDTH)
        h = max(h, self.MIN_HEIGHT)
        
        # Ensure position is on screen
        screen = QApplication.primaryScreen().geometry()
        x = max(0, min(x, screen.width() - w))
        y = max(0, min(y, screen.height() - h))
        
        return [x, y, w, h]
        
        # Update corner positions
        self._update_corner_positions()
    
    def apply_style(self):
        """Glassmorphic professional styling"""
        fs = config.get("font_size", 20)
        tc = config.get("text_color", "#FFFFFF")
        bg = config.get("bg_color", "rgba(20,20,30,0.90)")
        
        self.setStyleSheet(f"""
            #overlayContainer {{
                background: {bg};
                border-radius: 16px;
                border: 2px solid rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
            }}
            QLabel {{
                color: {tc};
                font-size: {fs}px;
                font-weight: 400;
                line-height: 1.6;
                background: transparent;
            }}
        """)
    
    def _update_corner_positions(self):
        """Update visual corner indicators"""
        w, h = self.width(), self.height()
        positions = [
            (5, 5),  # Top-left
            (w - 13, 5),  # Top-right
            (5, h - 13),  # Bottom-left
            (w - 13, h - 13)  # Bottom-right
        ]
        for indicator, pos in zip(self.corner_indicators, positions):
            indicator.move(*pos)
    
    def add_text(self, text, is_translation=True):
        """Add text with Google Live Captions-style smooth scrolling and instant updates"""
        if not text.strip():
            return
        
        if not self.live_captions_mode:
            # Legacy mode: just add to buffer
            for word in text.split():
                self.text_buffer.append(word)
            self.update_display_legacy()
            return
        
        # Google Live Captions mode: instant word-by-word updates
        words = text.split()
        
        # Add words one at a time for smooth appearance (if delay is small enough)
        if self.subtitle_update_delay < 0.020:  # Less than 20ms
            # Ultra-fast mode: add all words at once
            self.current_sentence.extend(words)
            self.update_display_smooth()
        else:
            # Smooth mode: add words with slight delay for visual effect
            for word in words:
                self.current_sentence.append(word)
                self.update_display_smooth()
                QApplication.processEvents()  # Allow UI to update
        
        # Check if we should create a new line (sentence break detection)
        sentence_text = " ".join(self.current_sentence)
        should_new_line = (
            len(self.current_sentence) > 12 or  # Wrap sooner for readability
            any(sentence_text.rstrip().endswith(punct) for punct in ['.', '!', '?', '。', '！', '？', ',', ';'])
        )
        
        if should_new_line and len(self.current_sentence) > 0:
            # Move current sentence to displayed lines with smooth transition
            self.displayed_lines.append(sentence_text)
            self.current_sentence = []
            self.update_display_smooth()
    
    def update_display_legacy(self):
        """Legacy update mode (simple buffer display)"""
        if not self.text_buffer:
            self.label.setText("")
            return
        text = " ".join(list(self.text_buffer))
        self.label.setText(text)
    
    def update_display_smooth(self):
        """Update display with smooth transitions like Google Live Captions - optimized for speed"""
        if not self.displayed_lines and not self.current_sentence:
            self.label.setText("")
            return
        
        # Build display text: show last N completed lines + current building sentence
        max_lines = config.get("subtitle_lines", 3)
        lines = list(self.displayed_lines)
        if self.current_sentence:
            lines.append(" ".join(self.current_sentence))
        
        # Keep only last N lines for clean display (Google Live Captions style)
        display_text = "\n".join(lines[-max_lines:])
        
        # Update with configurable throttling for live effect
        current_time = time.time()
        time_since_last = current_time - self.last_update_time
        
        # Use configurable delay (default 10ms = 100 FPS for very responsive updates)
        if time_since_last > self.subtitle_update_delay:
            self.label.setText(display_text)
            # Subtle fade for new content (only on longer pauses)
            if self.fade_animation.state() != QPropertyAnimation.State.Running and time_since_last > 0.3:
                self.fade_animation.setStartValue(0.85)
                self.fade_animation.setEndValue(1.0)
                self.fade_animation.setDuration(100)  # Faster animation
                self.fade_animation.start()
            self.last_update_time = current_time
        else:
            # For very rapid updates, just update text without animation for maximum speed
            self.label.setText(display_text)
    
    def clear_subtitles(self):
        """Clear all subtitle text"""
        self.text_buffer.clear()
        self.displayed_lines.clear()
        self.current_sentence = []
        self.label.setText("")
    
    def _get_resize_mode(self, pos):
        """Determine resize mode based on cursor position"""
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        cs, es = self.CORNER_SIZE, self.EDGE_SIZE
        
        # Check corners first (priority)
        if x < cs and y < cs:
            return self.ResizeMode.TOP_LEFT
        elif x > w - cs and y < cs:
            return self.ResizeMode.TOP_RIGHT
        elif x < cs and y > h - cs:
            return self.ResizeMode.BOTTOM_LEFT
        elif x > w - cs and y > h - cs:
            return self.ResizeMode.BOTTOM_RIGHT
        
        # Check edges
        elif y < es:
            return self.ResizeMode.TOP
        elif y > h - es:
            return self.ResizeMode.BOTTOM
        elif x < es:
            return self.ResizeMode.LEFT
        elif x > w - es:
            return self.ResizeMode.RIGHT
        
        # Center = move
        return self.ResizeMode.MOVE
    
    def _update_cursor(self, mode):
        """Update cursor based on resize mode"""
        cursor_map = {
            self.ResizeMode.NONE: Qt.CursorShape.ArrowCursor,
            self.ResizeMode.MOVE: Qt.CursorShape.SizeAllCursor,
            self.ResizeMode.TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
            self.ResizeMode.BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
            self.ResizeMode.TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
            self.ResizeMode.BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
            self.ResizeMode.TOP: Qt.CursorShape.SizeVerCursor,
            self.ResizeMode.BOTTOM: Qt.CursorShape.SizeVerCursor,
            self.ResizeMode.LEFT: Qt.CursorShape.SizeHorCursor,
            self.ResizeMode.RIGHT: Qt.CursorShape.SizeHorCursor,
        }
        self.setCursor(cursor_map.get(mode, Qt.CursorShape.ArrowCursor))
    
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.resize_mode = self._get_resize_mode(e.pos())
            self.drag_start_pos = e.globalPosition().toPoint()
            self.drag_start_geometry = self.geometry()
            
            # Show corner indicators
            for indicator in self.corner_indicators:
                indicator.show()
    
    def mouseMoveEvent(self, e):
        if self.resize_mode == self.ResizeMode.NONE:
            # Update cursor for hover
            mode = self._get_resize_mode(e.pos())
            self._update_cursor(mode)
            return
        
        if not self.drag_start_pos:
            return
        
        delta = e.globalPosition().toPoint() - self.drag_start_pos
        geom = QRect(self.drag_start_geometry)
        
        # Apply resize/move based on mode
        if self.resize_mode == self.ResizeMode.MOVE:
            geom.moveTopLeft(geom.topLeft() + delta)
        
        elif self.resize_mode == self.ResizeMode.TOP_LEFT:
            geom.setTopLeft(geom.topLeft() + delta)
        elif self.resize_mode == self.ResizeMode.TOP_RIGHT:
            geom.setTopRight(geom.topRight() + delta)
        elif self.resize_mode == self.ResizeMode.BOTTOM_LEFT:
            geom.setBottomLeft(geom.bottomLeft() + delta)
        elif self.resize_mode == self.ResizeMode.BOTTOM_RIGHT:
            geom.setBottomRight(geom.bottomRight() + delta)
        
        elif self.resize_mode == self.ResizeMode.TOP:
            geom.setTop(geom.top() + delta.y())
        elif self.resize_mode == self.ResizeMode.BOTTOM:
            geom.setBottom(geom.bottom() + delta.y())
        elif self.resize_mode == self.ResizeMode.LEFT:
            geom.setLeft(geom.left() + delta.x())
        elif self.resize_mode == self.ResizeMode.RIGHT:
            geom.setRight(geom.right() + delta.x())
        
        # Enforce minimum size with more flexible constraints
        min_w = max(self.MIN_WIDTH, 135)
        min_h = max(self.MIN_HEIGHT, 57)
        
        if geom.width() < min_w:
            if self.resize_mode in [self.ResizeMode.LEFT, self.ResizeMode.TOP_LEFT, self.ResizeMode.BOTTOM_LEFT]:
                geom.setLeft(geom.right() - min_w)
            else:
                geom.setWidth(min_w)
        
        if geom.height() < min_h:
            if self.resize_mode in [self.ResizeMode.TOP, self.ResizeMode.TOP_LEFT, self.ResizeMode.TOP_RIGHT]:
                geom.setTop(geom.bottom() - min_h)
            else:
                geom.setHeight(min_h)
        
        # Validate geometry before setting to avoid Windows errors
        if geom.width() >= self.MIN_WIDTH and geom.height() >= self.MIN_HEIGHT:
            try:
                self.setGeometry(geom)
            except Exception as e:
                log.debug(f"Geometry set failed: {e}")
                # Fall back to current geometry if setting fails
                pass
        self._update_corner_positions()
    
    def mouseReleaseEvent(self, e):
        if self.resize_mode != self.ResizeMode.NONE:
            geom = self.geometry()
            # Validate and save geometry
            validated_geom = self._validate_geometry([geom.x(), geom.y(), geom.width(), geom.height()])
            config.set("overlay_position", validated_geom)
            
            # Hide corner indicators
            for indicator in self.corner_indicators:
                indicator.hide()
        
        self.resize_mode = self.ResizeMode.NONE
        self.drag_start_pos = None
        self.drag_start_geometry = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.container.setGeometry(0, 0, self.width(), self.height())
        self._update_corner_positions()
    
    def enterEvent(self, e):
        # Show corner indicators on hover
        for indicator in self.corner_indicators:
            indicator.show()
    
    def leaveEvent(self, e):
        # Hide corner indicators when not hovering
        if self.resize_mode == self.ResizeMode.NONE:
            for indicator in self.corner_indicators:
                indicator.hide()

class SettingsDialog(QDialog):
    """Comprehensive settings dialog with all editable options"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Settings")
        self.setMinimumWidth(600)
        self.setMinimumHeight(700)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tab widget for organized settings
        tabs = QTabWidget()
        
        # ===== GENERAL TAB =====
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)
        
        # Theme
        theme_group = QGroupBox("Appearance")
        theme_layout = QVBoxLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setCurrentText(config.get("theme", "dark").title())
        theme_layout.addWidget(QLabel("Theme:"))
        theme_layout.addWidget(self.theme_combo)
        
        # Font size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 48)
        self.font_size_spin.setValue(config.get("font_size", 20))
        theme_layout.addWidget(QLabel("Overlay Font Size:"))
        theme_layout.addWidget(self.font_size_spin)
        
        # Text color
        self.text_color_btn = QPushButton("Choose Text Color")
        self.text_color = config.get("text_color", "#FFFFFF")
        self.text_color_btn.clicked.connect(self.choose_text_color)
        theme_layout.addWidget(QLabel("Overlay Text Color:"))
        theme_layout.addWidget(self.text_color_btn)
        
        theme_group.setLayout(theme_layout)
        general_layout.addWidget(theme_group)
        
        # Animation
        anim_group = QGroupBox("Animation")
        anim_layout = QVBoxLayout()
        self.anim_duration_spin = QSpinBox()
        self.anim_duration_spin.setRange(0, 1000)
        self.anim_duration_spin.setValue(config.get("animation_duration", 200))
        self.anim_duration_spin.setSuffix(" ms")
        anim_layout.addWidget(QLabel("Animation Duration:"))
        anim_layout.addWidget(self.anim_duration_spin)
        anim_group.setLayout(anim_layout)
        general_layout.addWidget(anim_group)
        
        # Max words and subtitle settings
        words_group = QGroupBox("Overlay & Subtitle Settings")
        words_layout = QVBoxLayout()
        
        self.max_words_spin = QSpinBox()
        self.max_words_spin.setRange(10, 500)
        self.max_words_spin.setValue(config.get("max_words", 100))
        words_layout.addWidget(QLabel("Maximum Words in Overlay:"))
        words_layout.addWidget(self.max_words_spin)
        
        # Subtitle lines
        self.subtitle_lines_spin = QSpinBox()
        self.subtitle_lines_spin.setRange(1, 10)
        self.subtitle_lines_spin.setValue(config.get("subtitle_lines", 3))
        words_layout.addWidget(QLabel("Number of Subtitle Lines:"))
        words_layout.addWidget(self.subtitle_lines_spin)
        
        # Subtitle update speed
        self.subtitle_delay_spin = QSpinBox()
        self.subtitle_delay_spin.setRange(1, 100)
        self.subtitle_delay_spin.setValue(config.get("subtitle_update_delay", 10))
        self.subtitle_delay_spin.setSuffix(" ms")
        words_layout.addWidget(QLabel("Subtitle Update Speed (lower = faster):"))
        words_layout.addWidget(self.subtitle_delay_spin)
        
        speed_hint = QLabel("💡 Tip: 5-10ms for instant updates, 30-50ms for smoother animation")
        speed_hint.setWordWrap(True)
        speed_hint.setStyleSheet("color: #888; font-size: 11px;")
        words_layout.addWidget(speed_hint)
        
        words_group.setLayout(words_layout)
        general_layout.addWidget(words_group)
        
        general_layout.addStretch()
        tabs.addTab(general_tab, "General")
        
        # ===== AUDIO TAB =====
        audio_tab = QWidget()
        audio_layout = QVBoxLayout(audio_tab)
        
        # TTS Settings
        tts_group = QGroupBox("Text-to-Speech")
        tts_layout = QVBoxLayout()
        
        self.tts_rate_spin = QSpinBox()
        self.tts_rate_spin.setRange(50, 300)
        self.tts_rate_spin.setValue(config.get("tts_rate", 150))
        tts_layout.addWidget(QLabel("Speech Rate:"))
        tts_layout.addWidget(self.tts_rate_spin)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(config.get("volume", 0.8) * 100))
        self.volume_label = QLabel(f"{self.volume_slider.value()}%")
        self.volume_slider.valueChanged.connect(lambda v: self.volume_label.setText(f"{v}%"))
        tts_layout.addWidget(QLabel("Volume:"))
        tts_layout.addWidget(self.volume_slider)
        tts_layout.addWidget(self.volume_label)
        
        tts_group.setLayout(tts_layout)
        audio_layout.addWidget(tts_group)
        
        # Audio Device
        device_group = QGroupBox("Audio Input Device")
        device_layout = QVBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.addItem("Default", "default")
        for device in audio_device_manager.input_devices:
            self.device_combo.addItem(f"{device.name} ({device.host_api})", str(device.index))
        current_device = config.get("audio_device_input", "default")
        idx = self.device_combo.findData(current_device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        device_layout.addWidget(QLabel("Input Device:"))
        device_layout.addWidget(self.device_combo)
        device_group.setLayout(device_layout)
        audio_layout.addWidget(device_group)
        
        audio_layout.addStretch()
        tabs.addTab(audio_tab, "Audio")
        
        # ===== GPU/ENGINE TAB =====
        engine_tab = QWidget()
        engine_layout = QVBoxLayout(engine_tab)
        
        # GPU Settings
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QVBoxLayout()
        
        self.use_gpu_check = QCheckBox("Enable GPU Acceleration (CUDA/MPS)")
        self.use_gpu_check.setChecked(config.get("use_gpu", True))
        self.use_gpu_check.setEnabled(gpu_manager.is_gpu_available())
        gpu_layout.addWidget(self.use_gpu_check)
        
        gpu_info = QLabel(f"Status: {gpu_manager.device_name}")
        gpu_info.setWordWrap(True)
        gpu_layout.addWidget(gpu_info)
        
        if not gpu_manager.is_gpu_available():
            gpu_warning = QLabel("⚠️ No GPU detected. Install CUDA (NVIDIA) or use Apple Silicon for GPU acceleration.")
            gpu_warning.setWordWrap(True)
            gpu_warning.setStyleSheet("color: #FFC107;")
            gpu_layout.addWidget(gpu_warning)
        
        gpu_group.setLayout(gpu_layout)
        engine_layout.addWidget(gpu_group)
        
        # Whisper Model
        whisper_group = QGroupBox("Whisper Model")
        whisper_layout = QVBoxLayout()
        self.whisper_model_combo = QComboBox()
        for model in WHISPER_MODELS:
            self.whisper_model_combo.addItem(model.title(), model)
        current_model = config.get("whisper_model", "base")
        idx = self.whisper_model_combo.findData(current_model)
        if idx >= 0:
            self.whisper_model_combo.setCurrentIndex(idx)
        whisper_layout.addWidget(QLabel("Model Size:"))
        whisper_layout.addWidget(self.whisper_model_combo)
        
        model_info = QLabel(
            "• Tiny: Fastest, less accurate\\n"
            "• Base: Balanced (recommended)\\n"
            "• Small/Medium/Large: Higher accuracy, slower"
        )
        model_info.setWordWrap(True)
        whisper_layout.addWidget(model_info)
        whisper_group.setLayout(whisper_layout)
        engine_layout.addWidget(whisper_group)
        
        engine_layout.addStretch()
        tabs.addTab(engine_tab, "GPU & Models")
        
        # ===== CACHE TAB =====
        cache_tab = QWidget()
        cache_layout = QVBoxLayout(cache_tab)
        
        cache_group = QGroupBox("Translation Cache")
        cache_group_layout = QVBoxLayout()
        
        self.cache_enabled = QCheckBox("Enable Translation Caching")
        self.cache_enabled.setChecked(config.get("cache_translations", True))
        cache_group_layout.addWidget(self.cache_enabled)
        
        self.cache_expiry_spin = QSpinBox()
        self.cache_expiry_spin.setRange(1, 365)
        self.cache_expiry_spin.setValue(config.get("cache_expiry_days", 7))
        self.cache_expiry_spin.setSuffix(" days")
        cache_group_layout.addWidget(QLabel("Cache Expiry:"))
        cache_group_layout.addWidget(self.cache_expiry_spin)
        
        cache_group.setLayout(cache_group_layout)
        cache_layout.addWidget(cache_group)
        
        # Data locations
        data_group = QGroupBox("Data Locations")
        data_layout = QVBoxLayout()
        data_layout.addWidget(QLabel(f"Configuration: {CONFIG_FILE}"))
        data_layout.addWidget(QLabel(f"Database: {DB_FILE}"))
        data_layout.addWidget(QLabel(f"Audio Cache: {AUDIO_DIR}"))
        data_layout.addWidget(QLabel(f"Vosk Models: {VOSK_MODELS_DIR}"))
        data_group.setLayout(data_layout)
        cache_layout.addWidget(data_group)
        
        cache_layout.addStretch()
        tabs.addTab(cache_tab, "Cache & Data")
        
        layout.addWidget(tabs)
        
        # Buttons
        buttons = QHBoxLayout()
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.save_settings)
        cancel_btn = QPushButton("✖ Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addStretch()
        buttons.addWidget(save_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)
    
    def choose_text_color(self):
        color = QColorDialog.getColor(QColor(self.text_color), self, "Choose Text Color")
        if color.isValid():
            self.text_color = color.name()
            self.text_color_btn.setStyleSheet(f"background-color: {self.text_color};")
    
    def save_settings(self):
        # Save all settings
        config.set("theme", self.theme_combo.currentText().lower())
        config.set("font_size", self.font_size_spin.value())
        config.set("text_color", self.text_color)
        config.set("animation_duration", self.anim_duration_spin.value())
        config.set("max_words", self.max_words_spin.value())
        config.set("subtitle_lines", self.subtitle_lines_spin.value())
        config.set("subtitle_update_delay", self.subtitle_delay_spin.value())
        config.set("tts_rate", self.tts_rate_spin.value())
        config.set("volume", self.volume_slider.value() / 100.0)
        config.set("audio_device_input", self.device_combo.currentData())
        config.set("use_gpu", self.use_gpu_check.isChecked())
        config.set("whisper_model", self.whisper_model_combo.currentData())
        config.set("cache_translations", self.cache_enabled.isChecked())
        config.set("cache_expiry_days", self.cache_expiry_spin.value())
        
        # Apply TTS volume immediately
        tts_manager.set_volume(self.volume_slider.value() / 100.0)
        
        # Apply GPU setting immediately
        whisper_manager.set_device(self.use_gpu_check.isChecked())
        
        self.accept()

class ModelManagerDialog(QDialog):
    """Model download and management dialog"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📥 Model Manager")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        tabs = QTabWidget()
        
        # ===== WHISPER TAB =====
        whisper_tab = QWidget()
        whisper_layout = QVBoxLayout(whisper_tab)
        
        whisper_info = QLabel(
            "Whisper models download automatically on first use.\\n"
            "Models are cached in your home directory (~/.cache/whisper)."
        )
        whisper_info.setWordWrap(True)
        whisper_layout.addWidget(whisper_info)
        
        whisper_group = QGroupBox("Available Whisper Models")
        whisper_group_layout = QVBoxLayout()
        
        for model in WHISPER_MODELS:
            model_row = QHBoxLayout()
            label = QLabel(f"{model.title()}")
            model_row.addWidget(label)
            
            size_label = QLabel()
            if model == "tiny":
                size_label.setText("~75 MB")
            elif model == "base":
                size_label.setText("~150 MB")
            elif model == "small":
                size_label.setText("~500 MB")
            elif model == "medium":
                size_label.setText("~1.5 GB")
            elif model == "large":
                size_label.setText("~3 GB")
            model_row.addWidget(size_label)
            
            model_row.addStretch()
            
            download_btn = QPushButton("📥 Download")
            download_btn.clicked.connect(lambda checked, m=model: self.download_whisper_model(m))
            model_row.addWidget(download_btn)
            
            whisper_group_layout.addLayout(model_row)
        
        whisper_group.setLayout(whisper_group_layout)
        whisper_layout.addWidget(whisper_group)
        
        current_device = QLabel(f"Current Device: {whisper_manager.device}")
        whisper_layout.addWidget(current_device)
        
        whisper_layout.addStretch()
        tabs.addTab(whisper_tab, "Whisper Models")
        
        # ===== VOSK TAB =====
        vosk_tab = QWidget()
        vosk_layout = QVBoxLayout(vosk_tab)
        
        vosk_info = QLabel(
            f"Vosk models are downloaded to: {VOSK_MODELS_DIR}\\n"
            "These are offline speech recognition models."
        )
        vosk_info.setWordWrap(True)
        vosk_layout.addWidget(vosk_info)
        
        vosk_group = QGroupBox("Available Vosk Models")
        vosk_group_layout = QVBoxLayout()
        
        for lang_code, url in VOSK_MODELS.items():
            model_row = QHBoxLayout()
            
            lang_name = {"en": "English", "fr": "French", "es": "Spanish", "de": "German"}.get(lang_code, lang_code)
            label = QLabel(f"{lang_name} ({lang_code})")
            model_row.addWidget(label)
            
            has_model = vosk_manager.has_model(lang_code)
            status = QLabel("✅ Installed" if has_model else "❌ Not installed")
            model_row.addWidget(status)
            
            model_row.addStretch()
            
            if not has_model:
                download_btn = QPushButton("📥 Download (~50 MB)")
                download_btn.clicked.connect(lambda checked, lc=lang_code: self.download_vosk_model(lc))
                model_row.addWidget(download_btn)
            
            vosk_group_layout.addLayout(model_row)
        
        vosk_group.setLayout(vosk_group_layout)
        vosk_layout.addWidget(vosk_group)
        
        vosk_layout.addStretch()
        tabs.addTab(vosk_tab, "Vosk Models")
        
        layout.addWidget(tabs)
        
        # Close button
        close_btn = QPushButton("✖ Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
    
    def download_whisper_model(self, model_name):
        reply = QMessageBox.question(
            self,
            "Download Whisper Model",
            f"Download {model_name} model?\\n\\nThe model will download automatically on first use.\\n"
            f"You can test it by selecting '{model_name}' in Settings and using Whisper engine.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Create progress dialog
            progress = QProgressDialog(f"Downloading Whisper {model_name} model...", None, 0, 0, self)
            progress.setWindowTitle("Downloading")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            QApplication.processEvents()
            
            # Download in thread to avoid blocking UI
            success = whisper_manager.load_model(model_name)
            
            progress.close()
            
            if success:
                QMessageBox.information(self, "Success", f"Whisper {model_name} model loaded successfully!")
            else:
                QMessageBox.warning(self, "Error", f"Failed to download {model_name} model. Check logs for details.")
    
    def download_vosk_model(self, lang_code):
        reply = QMessageBox.question(
            self,
            "Download Vosk Model",
            f"Download Vosk model for {lang_code}?\\n\\nSize: ~50 MB",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            progress = QProgressDialog(f"Downloading Vosk {lang_code} model...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Downloading")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            def update_progress(pct):
                progress.setValue(int(pct * 100))
                QApplication.processEvents()
            
            success = vosk_manager.download_model(lang_code, update_progress)
            
            progress.close()
            
            if success:
                QMessageBox.information(self, "Success", f"Vosk {lang_code} model downloaded successfully!")
                # Refresh the dialog
                self.close()
                new_dialog = ModelManagerDialog(self.parent())
                new_dialog.exec()
            else:
                QMessageBox.warning(self, "Error", f"Failed to download {lang_code} model. Check logs for details.")

class LiveTranslatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌍 Universal Live Translator — Professional Edition v3.0")
        self.resize(1200, 950)
        self.recognition_thread = None
        self.source_type = "microphone"
        self.translation_mode = "Continuous"
        self.recognition_engine = RecognitionEngine.GOOGLE
        self.overlay = ResizableOverlay()
        self.overlay.show()
        
        # Performance monitoring
        self.last_history_refresh = time.time()
        self.history_refresh_throttle = 2.0  # seconds
        
        self.setup_ui()
        self.apply_theme(config.get("theme", "dark"))
        self.load_settings()
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(15)
        
        # Top bar with GPU status
        top_bar = QHBoxLayout()
        self.status_label = QLabel("🌐 Ready")
        top_bar.addWidget(self.status_label)
        
        # GPU status indicator
        gpu_color = "#4CAF50" if gpu_manager.is_gpu_available() else "#FFC107"
        self.gpu_status = QLabel(f"🚀 {gpu_manager.device_name}")
        self.gpu_status.setStyleSheet(f"color: {gpu_color}; font-weight: bold;")
        self.gpu_status.setToolTip(f"Device: {gpu_manager.device}\nCUDA Available: {gpu_manager.has_cuda}\nMPS Available: {gpu_manager.has_mps}")
        top_bar.addWidget(self.gpu_status)
        
        # Performance monitor
        self.perf_label = QLabel("⚡ Performance: Ready")
        top_bar.addWidget(self.perf_label)
        
        top_bar.addStretch()
        
        help_btn = QPushButton("❓ F1")
        help_btn.clicked.connect(self.show_help)
        top_bar.addWidget(help_btn)
        
        self.models_btn = QPushButton("📥 Models")
        self.models_btn.clicked.connect(self.show_models)
        self.models_btn.setToolTip("Download and manage Whisper and Vosk models")
        top_bar.addWidget(self.models_btn)
        
        self.theme_btn = QPushButton("🌓 Theme")
        self.theme_btn.clicked.connect(self.toggle_theme)
        top_bar.addWidget(self.theme_btn)
        
        self.settings_btn = QPushButton("⚙️ Settings")
        self.settings_btn.clicked.connect(self.show_settings)
        self.settings_btn.setToolTip("Configure all application settings")
        top_bar.addWidget(self.settings_btn)
        
        main_layout.addLayout(top_bar)
        
        # Config with GPU toggle
        config_group = QGroupBox("🌐 Configuration")
        config_layout = QVBoxLayout()
        
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("From:"))
        self.source_lang_combo = QComboBox()
        self.populate_languages(self.source_lang_combo, True)
        lang_row.addWidget(self.source_lang_combo, 1)
        
        self.swap_btn = QPushButton("⇄")
        self.swap_btn.setMaximumWidth(50)
        self.swap_btn.clicked.connect(self.swap_languages)
        lang_row.addWidget(self.swap_btn)
        
        lang_row.addWidget(QLabel("To:"))
        self.target_lang_combo = QComboBox()
        self.populate_languages(self.target_lang_combo)
        lang_row.addWidget(self.target_lang_combo, 1)
        
        config_layout.addLayout(lang_row)
        
        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("Engine:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("🌐 Google", RecognitionEngine.GOOGLE.value)
        self.engine_combo.addItem("💻 Vosk", RecognitionEngine.VOSK.value)
        self.engine_combo.addItem("🤖 Whisper", RecognitionEngine.WHISPER.value)
        self.engine_combo.currentIndexChanged.connect(self.on_engine_changed)
        engine_row.addWidget(self.engine_combo, 1)
        
        engine_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Continuous", "Real-time", "Manual"])
        self.mode_combo.currentTextChanged.connect(lambda m: setattr(self, 'translation_mode', m))
        engine_row.addWidget(self.mode_combo, 1)
        
        config_layout.addLayout(engine_row)
        
        # GPU toggle
        gpu_row = QHBoxLayout()
        self.gpu_checkbox = QCheckBox("🚀 Use GPU Acceleration")
        self.gpu_checkbox.setChecked(config.get("use_gpu", True))
        self.gpu_checkbox.setEnabled(gpu_manager.is_gpu_available())
        self.gpu_checkbox.stateChanged.connect(self.on_gpu_toggled)
        gpu_row.addWidget(self.gpu_checkbox)
        
        if not gpu_manager.is_gpu_available():
            gpu_row.addWidget(QLabel("(No GPU detected)"))
        
        gpu_row.addStretch()
        config_layout.addLayout(gpu_row)
        
        self.model_status = QLabel("Ready")
        config_layout.addWidget(self.model_status)
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # Input/Output
        io_splitter = QSplitter(Qt.Orientation.Vertical)
        
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 0, 0)
        
        input_header = QHBoxLayout()
        input_header.addWidget(QLabel("🎙️ Input (Continuous)"))
        self.confidence_label = QLabel()
        input_header.addWidget(self.confidence_label)
        input_header.addStretch()
        self.word_count = QLabel("0 words")
        input_header.addWidget(self.word_count)
        input_layout.addLayout(input_header)
        
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("🎤 Speak continuously - no need to stop...")
        self.input_text.textChanged.connect(self.update_word_count)
        input_layout.addWidget(self.input_text)
        
        io_splitter.addWidget(input_widget)
        
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        
        output_header = QHBoxLayout()
        output_header.addWidget(QLabel("🈯 Translation (Async Pipeline)"))
        output_header.addStretch()
        self.copy_btn = QPushButton("📋 Copy")
        self.copy_btn.clicked.connect(self.copy_output)
        output_header.addWidget(self.copy_btn)
        output_layout.addLayout(output_header)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("Translation appears here instantly...")
        output_layout.addWidget(self.output_text)
        
        io_splitter.addWidget(output_widget)
        main_layout.addWidget(io_splitter, 3)
        
        # Controls
        controls = QWidget()
        controls_layout = QGridLayout(controls)
        
        self.listen_btn = QPushButton("🎤 Start Continuous Listening")
        self.listen_btn.clicked.connect(self.toggle_listening)
        self.listen_btn.setStyleSheet("font-weight:bold;padding:12px;font-size:14px;background:#4CAF50;color:white;")
        self.listen_btn.setToolTip("Start continuous speech recognition (Ctrl+L)")
        controls_layout.addWidget(self.listen_btn, 0, 0, 1, 2)
        
        self.translate_btn = QPushButton("🔄 Translate")
        self.translate_btn.clicked.connect(self.translate_manual)
        self.translate_btn.setToolTip("Manually translate the input text (Ctrl+T)")
        controls_layout.addWidget(self.translate_btn, 0, 2, 1, 2)
        
        self.speak_btn = QPushButton("🔊 Speak")
        self.speak_btn.clicked.connect(self.speak_output)
        self.speak_btn.setToolTip("Speak the translated text (Ctrl+S)")
        controls_layout.addWidget(self.speak_btn, 1, 0)
        
        self.stop_btn = QPushButton("🔇 Stop")
        self.stop_btn.clicked.connect(lambda: tts_manager.stop())
        controls_layout.addWidget(self.stop_btn, 1, 1)
        
        self.source_btn = QPushButton("🎧 System Audio")
        self.source_btn.clicked.connect(self.toggle_source)
        self.source_btn.setToolTip("Switch between microphone and system audio input")
        controls_layout.addWidget(self.source_btn, 1, 2)
        
        self.overlay_btn = QPushButton("👁️ Overlay")
        self.overlay_btn.clicked.connect(self.toggle_overlay)
        self.overlay_btn.setToolTip("Toggle the translation overlay (Ctrl+O)")
        controls_layout.addWidget(self.overlay_btn, 1, 3)
        
        self.auto_speak = QCheckBox("Auto-speak")
        self.auto_speak.setChecked(config.get("auto_speak", True))
        self.auto_speak.stateChanged.connect(lambda: config.set("auto_speak", self.auto_speak.isChecked()))
        controls_layout.addWidget(self.auto_speak, 2, 0, 1, 2)
        
        self.show_conf = QCheckBox("Show confidence")
        self.show_conf.setChecked(config.get("show_confidence", True))
        self.show_conf.stateChanged.connect(lambda: config.set("show_confidence", self.show_conf.isChecked()))
        controls_layout.addWidget(self.show_conf, 2, 2, 1, 2)
        
        main_layout.addWidget(controls)
        
        # History
        history_group = QGroupBox("📜 History (Throttled Refresh)")
        history_layout = QVBoxLayout()
        
        history_controls = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.textChanged.connect(self.search_history)
        history_controls.addWidget(self.search_input)
        
        self.export_btn = QPushButton("💾 Export")
        self.export_btn.clicked.connect(self.export_history)
        history_controls.addWidget(self.export_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self.clear_history)
        history_controls.addWidget(self.clear_btn)
        
        history_layout.addLayout(history_controls)
        
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self.load_history_item)
        history_layout.addWidget(self.history_list)
        
        history_group.setLayout(history_layout)
        main_layout.addWidget(history_group, 1)
        
        self.statusBar().showMessage("Ready - Professional Edition v3.0")
        self.refresh_history()
        self.setup_shortcuts()
    
    def setup_shortcuts(self):
        QShortcut(QKeySequence("F1"), self, self.show_help)
        QShortcut(QKeySequence("Ctrl+L"), self, self.toggle_listening)
        QShortcut(QKeySequence("Ctrl+T"), self, self.translate_manual)
        QShortcut(QKeySequence("Ctrl+S"), self, self.speak_output)
        QShortcut(QKeySequence("Ctrl+O"), self, self.toggle_overlay)
        QShortcut(QKeySequence("Ctrl+D"), self, self.toggle_theme)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)
    
    def show_help(self):
        help_text = f"""
🎙️ Professional Edition v3.0 Features

✨ Continuous Listening
- Microphone never stops - speak naturally
- Async processing queue - no blocking
- Parallel recognition and translation

📺 Google Live Captions-Style Overlay
- Real-time word-by-word updates
- Smooth scrolling transitions
- Shows last 3 lines of text
- Auto-wrapping at sentence breaks
- Drag corners: Resize diagonally
- Drag edges: Resize in one direction  
- Drag center: Move window

🚀 GPU Acceleration (Enhanced)
- Current Status: {gpu_manager.device_name}
- Device: {gpu_manager.device}
- CUDA: {'✅ Available' if gpu_manager.has_cuda else '❌ Not detected'}
- 10-20x faster Whisper transcription
- Configure in Settings (⚙️)

📥 Model Management
- Click 'Models' to download Whisper/Vosk models
- Whisper: tiny, base, small, medium, large
- Vosk: Offline speech recognition

⚡ Keyboard Shortcuts
- F1: This help
- Ctrl+L: Toggle listening
- Ctrl+T: Manual translate
- Ctrl+S: Speak output
- Ctrl+O: Toggle overlay
- Ctrl+D: Toggle theme
- Ctrl+Q: Quit

⚙️ Settings
- All settings now editable in Settings dialog
- Audio devices, GPU, models, theme, etc.
- Changes apply immediately

🔧 Latest Fixes & Improvements:
- ✅ Enhanced system audio device detection with validation
- ✅ Detailed error messages for audio device issues
- ✅ Optimized subtitle overlay speed (configurable 1-100ms)
- ✅ Faster word-by-word subtitle updates
- ✅ Configurable subtitle lines and update delay
- ✅ Improved loopback device fallback mechanism
- ✅ Better error handling for PortAudio issues
- ✅ Instant subtitle mode for ultra-responsive captions
"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Help - Professional Edition v3.0")
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()
    
    def populate_languages(self, combo, include_auto=False):
        if include_auto:
            combo.addItem("🔍 Auto", "auto")
        langs = sorted([(n.title(), c) for n, c in GOOGLE_LANGUAGES_TO_CODES.items()], key=lambda x: x[0])
        for name, code in langs:
            combo.addItem(name, code)
    
    def load_settings(self):
        src = config.get("source_language", "auto")
        tgt = config.get("target_language", "fr")
        engine = config.get("recognition_engine", "google")
        idx = self.source_lang_combo.findData(src)
        if idx >= 0:
            self.source_lang_combo.setCurrentIndex(idx)
        idx = self.target_lang_combo.findData(tgt)
        if idx >= 0:
            self.target_lang_combo.setCurrentIndex(idx)
        idx = self.engine_combo.findData(engine)
        if idx >= 0:
            self.engine_combo.setCurrentIndex(idx)
    
    def on_engine_changed(self):
        engine_val = self.engine_combo.currentData()
        self.recognition_engine = RecognitionEngine(engine_val)
        config.set("recognition_engine", engine_val)
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.stop_listening()
            QTimer.singleShot(500, self.start_listening)
    
    def on_gpu_toggled(self):
        use_gpu = self.gpu_checkbox.isChecked()
        config.set("use_gpu", use_gpu)
        whisper_manager.set_device(use_gpu)
        device = whisper_manager.device
        self.gpu_status.setText(f"🚀 {device.upper()}")
        self.statusBar().showMessage(f"GPU {'enabled' if use_gpu else 'disabled'} - using {device}", 3000)
    
    def show_models(self):
        dialog = ModelManagerDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Refresh UI if needed
            pass
    
    def show_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Apply theme change if needed
            theme = config.get("theme", "dark")
            self.apply_theme(theme)
            # Update overlay style and configuration
            self.overlay.apply_style()
            # Update subtitle delay setting
            self.overlay.subtitle_update_delay = config.get("subtitle_update_delay", 10) / 1000.0
            # Recreate displayed_lines deque with new max lines
            max_lines = config.get("subtitle_lines", 3)
            old_lines = list(self.overlay.displayed_lines)
            self.overlay.displayed_lines = deque(old_lines, maxlen=max_lines)
            # Update GPU status
            device = whisper_manager.device
            self.gpu_status.setText(f"🚀 Device: {device.upper()}")
            self.gpu_checkbox.setChecked(config.get("use_gpu", True))
            self.statusBar().showMessage("Settings saved - Subtitle speed updated", 3000)
    
    def swap_languages(self):
        if self.source_lang_combo.currentData() == "auto":
            self.statusBar().showMessage("Cannot swap with auto-detect", 3000)
            return
        src_idx = self.source_lang_combo.currentIndex()
        tgt_idx = self.target_lang_combo.currentIndex()
        self.source_lang_combo.setCurrentIndex(tgt_idx)
        self.target_lang_combo.setCurrentIndex(src_idx - 1)
    
    def update_word_count(self):
        text = self.input_text.toPlainText()
        words = len(text.split()) if text.strip() else 0
        self.word_count.setText(f"{words} words")
    
    def toggle_listening(self):
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.stop_listening()
        else:
            self.start_listening()
    
    def start_listening(self):
        if self.recognition_thread:
            self.recognition_thread.stop()
            self.recognition_thread.wait()
        
        src_lang = self.source_lang_combo.currentData()
        if src_lang == "auto":
            src_lang = "en"
        
        device_idx = config.get("audio_device_input", "default")
        if device_idx == "default":
            device_idx = None
        else:
            device_idx = int(device_idx)
        
        # Use continuous recognition thread
        self.recognition_thread = ContinuousSpeechRecognitionThread(
            self.source_type, src_lang, self.recognition_engine, device_idx
        )
        self.recognition_thread.phrase_detected.connect(self.handle_phrase, Qt.ConnectionType.QueuedConnection)
        self.recognition_thread.status_changed.connect(self.update_status_safe, Qt.ConnectionType.QueuedConnection)
        self.recognition_thread.error_occurred.connect(
            self.handle_recognition_error, Qt.ConnectionType.QueuedConnection
        )
        self.recognition_thread.performance_update.connect(self.update_performance, Qt.ConnectionType.QueuedConnection)
        self.recognition_thread.start()
        
        self.listen_btn.setText("⏹️ Stop Continuous Listening")
        self.listen_btn.setStyleSheet("background:#D32F2F;color:white;font-weight:bold;padding:12px;font-size:14px;")
        self.statusBar().showMessage(f"🎙️ Continuous listening ({self.recognition_engine.value})...")
    
    def stop_listening(self):
        if self.recognition_thread:
            self.recognition_thread.stop()
            self.recognition_thread.wait()
            self.recognition_thread = None
        self.listen_btn.setText("🎤 Start Continuous Listening")
        self.listen_btn.setStyleSheet("font-weight:bold;padding:12px;font-size:14px;")
        self.statusBar().showMessage("Stopped")
        self.perf_label.setText("⚡ Performance: Idle")
    
    def toggle_source(self):
        self.source_type = "system" if self.source_type == "microphone" else "microphone"
        self.source_btn.setText("🎤 Microphone" if self.source_type == "system" else "🎧 System Audio")
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.stop_listening()
            QTimer.singleShot(500, self.start_listening)
    
    def update_status_safe(self, message):
        """Thread-safe status update"""
        QMetaObject.invokeMethod(
            self.statusBar(), "showMessage",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, message)
        )
    
    def handle_recognition_error(self, error_msg):
        """Thread-safe error handling"""
        QMessageBox.warning(self, "Error", error_msg)
        self.stop_listening()
    
    def update_performance(self, perf_data):
        """Update performance indicators (called from signal, already thread-safe)"""
        avg_time = perf_data.get('avg_processing_time', 0)
        rec_queue = perf_data.get('recognition_queue', 0)
        trans_queue = perf_data.get('translation_queue', 0)
        device = perf_data.get('device', 'N/A')
        
        self.perf_label.setText(
            f"⚡ {avg_time:.0f}ms | Q:{rec_queue}/{trans_queue} | {device}"
        )
    
    def handle_phrase(self, phrase, confidence):
        if not phrase.strip():
            return
        
        cursor = self.input_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        timestamp = datetime.now().strftime("%H:%M:%S")
        conf_str = f" [{confidence:.0%}]" if config.get("show_confidence") else ""
        cursor.insertText(f"[{timestamp}]{conf_str} {phrase}\n")
        self.input_text.setTextCursor(cursor)
        self.input_text.ensureCursorVisible()
        self.confidence_label.setText(f"Confidence: {confidence:.0%}")
        
        if self.translation_mode in ["Real-time", "Continuous"]:
            # Async translation - non-blocking
            self.translate_phrase_async(phrase, confidence)
    
    def translate_phrase_async(self, phrase, speech_conf=1.0):
        """Async translation - doesn't block recognition"""
        src_lang = self.source_lang_combo.currentData()
        tgt_lang = self.target_lang_combo.currentData()
        
        if src_lang == "auto":
            try:
                detections = detect_langs(phrase)
                src_lang = detections[0].lang if detections else "en"
            except:
                src_lang = "en"
        
        def on_translation_complete(result):
            translated, duration, trans_conf = result
            combined_conf = speech_conf * trans_conf
            timestamp = datetime.now().strftime("%H:%M:%S")
            conf_str = f" [{combined_conf:.0%}]" if config.get("show_confidence") else ""
            
            self.output_text.append(f"[{timestamp}]{conf_str} {translated}")
            self.overlay.add_text(translated)
            
            if config.get("auto_speak", True):
                tts_manager.speak(translated, tgt_lang)
            
            db.add_translation(phrase, translated, src_lang, tgt_lang, 
                             self.translation_mode, self.recognition_engine.value, 
                             combined_conf, duration)
            
            # Throttled history refresh
            self.refresh_history_throttled()
            
            self.statusBar().showMessage(f"✓ Translated ({duration}ms, {combined_conf:.0%})", 2000)
        
        # Submit async translation
        translator.translate_async(phrase, src_lang, tgt_lang, on_translation_complete)
    
    def translate_phrase(self, phrase, speech_conf=1.0):
        """Synchronous translation (for compatibility)"""
        src_lang = self.source_lang_combo.currentData()
        tgt_lang = self.target_lang_combo.currentData()
        
        if src_lang == "auto":
            try:
                detections = detect_langs(phrase)
                src_lang = detections[0].lang if detections else "en"
            except:
                src_lang = "en"
        
        translated, duration, trans_conf = translator.translate(phrase, src_lang, tgt_lang)
        combined_conf = speech_conf * trans_conf
        timestamp = datetime.now().strftime("%H:%M:%S")
        conf_str = f" [{combined_conf:.0%}]" if config.get("show_confidence") else ""
        
        self.output_text.append(f"[{timestamp}]{conf_str} {translated}")
        self.overlay.add_text(translated)
        
        if config.get("auto_speak", True):
            tts_manager.speak(translated, tgt_lang)
        
        db.add_translation(phrase, translated, src_lang, tgt_lang, 
                         self.translation_mode, self.recognition_engine.value, 
                         combined_conf, duration)
        
        self.refresh_history_throttled()
        self.statusBar().showMessage(f"Translated ({duration}ms, {combined_conf:.0%})", 2000)
    
    def translate_manual(self):
        text = self.input_text.toPlainText().strip()
        if not text:
            self.statusBar().showMessage("No text", 3000)
            return
        
        src_lang = self.source_lang_combo.currentData()
        tgt_lang = self.target_lang_combo.currentData()
        
        if src_lang == "auto":
            try:
                src_lang = detect_langs(text)[0].lang if detect_langs(text) else "en"
            except:
                src_lang = "en"
        
        self.statusBar().showMessage("Translating...")
        QApplication.processEvents()
        
        translated, duration, conf = translator.translate(text, src_lang, tgt_lang)
        self.output_text.clear()
        conf_str = f" [{conf:.0%}]" if config.get("show_confidence") else ""
        self.output_text.append(f"{translated}{conf_str}")
        self.overlay.add_text(translated)
        
        if config.get("auto_speak", True):
            tts_manager.speak(translated, tgt_lang)
        
        db.add_translation(text, translated, src_lang, tgt_lang, "Manual", "manual", conf, duration)
        self.refresh_history()
        self.statusBar().showMessage(f"Done ({duration}ms, {conf:.0%})", 3000)
    
    def speak_output(self):
        text = self.output_text.toPlainText().strip()
        if not text:
            return
        import re
        text = re.sub(r'\[.*?\]\s*', '', text)
        lang = self.target_lang_combo.currentData()
        tts_manager.speak(text, lang)
    
    def copy_output(self):
        text = self.output_text.toPlainText()
        import re
        text = re.sub(r'\[.*?\]\s*', '', text)
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage("Copied!", 2000)
    
    def toggle_overlay(self):
        self.overlay.setVisible(not self.overlay.isVisible())
    
    def refresh_history_throttled(self):
        """Throttled history refresh to avoid UI lag"""
        now = time.time()
        if now - self.last_history_refresh > self.history_refresh_throttle:
            self.refresh_history()
            self.last_history_refresh = now
    
    def refresh_history(self, search=""):
        self.history_list.clear()
        for row in db.get_history(50, search):
            try:
                dt = datetime.fromisoformat(row['timestamp'])
                time_str = dt.strftime("%H:%M")
            except:
                time_str = ""
            
            src_prev = row['source_text'][:40] + "..." if len(row['source_text']) > 40 else row['source_text']
            trans_prev = row['translated_text'][:40] + "..." if len(row['translated_text']) > 40 else row['translated_text']
            conf = row.get('confidence', 1.0)
            conf_str = f" [{conf:.0%}]" if conf < 1.0 else ""
            item_text = f"[{time_str}]{conf_str} {row['source_lang']}→{row['target_lang']} | {src_prev} ➜ {trans_prev}"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, row)
            
            if conf >= 0.9:
                item.setForeground(QColor("#4CAF50"))
            elif conf >= 0.7:
                item.setForeground(QColor("#FFC107"))
            else:
                item.setForeground(QColor("#FF5722"))
            
            self.history_list.addItem(item)
    
    def search_history(self):
        self.refresh_history(self.search_input.text())
    
    def load_history_item(self, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        if data:
            self.input_text.setPlainText(data['source_text'])
            self.output_text.setPlainText(data['translated_text'])
    
    def export_history(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export", 
            str(BASE_DIR / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"), 
            "Text Files (*.txt)"
        )
        if filename:
            try:
                db.export_history(filename)
                QMessageBox.information(self, "Success", f"Exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {e}")
    
    def clear_history(self):
        if QMessageBox.question(self, "Clear", "Clear all history?") == QMessageBox.StandardButton.Yes:
            db.clear_history()
            self.refresh_history()
    
    def toggle_theme(self):
        current = config.get("theme", "dark")
        new = "light" if current == "dark" else "dark"
        config.set("theme", new)
        self.apply_theme(new)
    
    def apply_theme(self, theme):
        if theme == "dark":
            # Professional dark theme with glassmorphic elements
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background: #1a1a1a;
                    color: #e0e0e0;
                }
                QGroupBox {
                    border: 2px solid #2d2d2d;
                    border-radius: 12px;
                    margin-top: 12px;
                    padding-top: 18px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #252525, stop:1 #1f1f1f);
                }
                QGroupBox::title {
                    left: 15px;
                    padding: 0 10px;
                    color: #64B5F6;
                    font-weight: bold;
                }
                QTextEdit, QListWidget, QLineEdit {
                    background: #252525;
                    border: 1px solid #3a3a3a;
                    border-radius: 8px;
                    padding: 10px;
                    selection-background-color: #64B5F6;
                }
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3a3a3a, stop:1 #2d2d2d);
                    border: 1px solid #4a4a4a;
                    border-radius: 8px;
                    padding: 10px 18px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4a4a4a, stop:1 #3d3d3d);
                    border: 1px solid #64B5F6;
                }
                QPushButton:pressed {
                    background: #2d2d2d;
                }
                QComboBox {
                    background: #252525;
                    border: 1px solid #3a3a3a;
                    border-radius: 8px;
                    padding: 8px;
                }
                QComboBox:hover {
                    border: 1px solid #64B5F6;
                }
                QCheckBox {
                    spacing: 8px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 4px;
                    border: 2px solid #3a3a3a;
                    background: #252525;
                }
                QCheckBox::indicator:checked {
                    background: #64B5F6;
                    border-color: #64B5F6;
                }
                QStatusBar {
                    background: #222;
                    border-top: 1px solid #3a3a3a;
                    color: #b0b0b0;
                }
            """)
        else:
            # Light theme
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background: #f5f5f5;
                    color: #212121;
                }
                QGroupBox {
                    border: 2px solid #e0e0e0;
                    border-radius: 12px;
                    background: white;
                }
                QPushButton {
                    background: white;
                    border: 1px solid #d0d0d0;
                    border-radius: 8px;
                    padding: 10px 18px;
                }
                QPushButton:hover {
                    border: 1px solid #2196F3;
                }
            """)
        
        # Update overlay style
        self.overlay.apply_style()
    
    def closeEvent(self, event):
        config.set("source_language", self.source_lang_combo.currentData())
        config.set("target_language", self.target_lang_combo.currentData())
        
        if self.recognition_thread:
            self.recognition_thread.stop()
            self.recognition_thread.wait()
        
        tts_manager.shutdown()
        audio_device_manager.cleanup()
        self.overlay.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Universal Live Translator Pro v3.0")
    app.setStyle("Fusion")
    
    log.info("="*60)
    log.info("Universal Live Translator — Professional Edition v3.0")
    log.info(f"Data: {BASE_DIR}")
    log.info(f"GPU: {gpu_manager.device_name}")
    log.info(f"Device: {gpu_manager.device}")
    log.info("="*60)
    
    window = LiveTranslatorApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
