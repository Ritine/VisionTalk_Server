import os
import shutil
import time
import subprocess
import tempfile
from typing import List, Tuple, Optional

import speech_recognition as sr
from gtts import gTTS
from qwen_runtime import generate

SAMPLE_MP3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "output_audio.mp3"))

# ASR override: set to an absolute path string to force using a fixed audio file for ASR
# ASR_OVERRIDE_AUDIO_PATH: Optional[str] = None
ASR_OVERRIDE_AUDIO_PATH = "/Users/ritine/Imperial/Indivisual_Project/server/static/test.mp3"


def _convert_to_wav_16k_mono(src_path: str) -> str:
    """Convert input audio to 16kHz mono WAV using ffmpeg. Returns path to temp wav."""
    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-ac", "1",  # mono
            "-ar", "16000",  # 16kHz
            tmp_wav
        ]
        # Suppress ffmpeg stdout/stderr
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return tmp_wav
    except Exception as e:
        # Cleanup file if created
        try:
            os.remove(tmp_wav)
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg convert failed: {e}")


def asr_transcribe(audio_path: str, language: str = "en-US") -> str:
    """Transcribe short audio (<60s). Converts to 16k mono WAV if needed, then uses Google Web Speech API.

    If ASR_OVERRIDE_AUDIO_PATH is set to an existing file path, that file will be
    used instead of the provided audio_path. This is useful to mock ASR for e2e testing.
    """
    override_path = ASR_OVERRIDE_AUDIO_PATH
    if override_path and os.path.exists(override_path):
        audio_path = override_path

    recognizer = sr.Recognizer()

    # Ensure WAV 16k mono for recognizer
    use_path = audio_path
    ext = os.path.splitext(audio_path)[1].lower()
    cleanup = None
    try:
        if ext not in {".wav", ".aiff", ".aif", ".aifc"}:
            use_path = _convert_to_wav_16k_mono(audio_path)
            cleanup = use_path
        with sr.AudioFile(use_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        return "[Unintelligible: speech not recognized]"
    except sr.RequestError as e:
        return f"[Error: API request failed: {e}]"
    except Exception as e:
        return f"[Error: {e}]"
    finally:
        if cleanup and os.path.exists(cleanup):
            try:
                os.remove(cleanup)
            except Exception:
                pass


def prepare_qwen_vl_inputs(transcript_text: str, frames: List[Tuple[int, str]], max_frames: int = 30) -> List[dict]:
    """Prepare messages for model without doing any visual encoding.

    Returns a messages list following the common multimodal chat schema where each
    content item can be of type 'image' (file path) or 'text'. This keeps the server
    decoupled from model runtime; the model process can directly consume these paths.
    """
    selected_paths = [path for _, path in frames[:max_frames]]

    user_content = []
    for img_path in selected_paths:
        user_content.append({"type": "image", "image": img_path})
    user_text = (
        # "ASR transcript (English may be present):\n"
        f"{transcript_text}\n\n"
        "Please answer concisely in English, using the images as context."
    )
    user_content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful multimodal assistant."}]},
        {"role": "user", "content": user_content},
    ]
    return messages


def multimodal_reason(transcript_text: str, frames: List[Tuple[int, str]]) -> str:
    messages = prepare_qwen_vl_inputs(transcript_text, frames)
    try:
        return generate(messages, max_new_tokens=64)
    except Exception:
        if not frames:
            return f"You said: {transcript_text}. No frames captured."
        first_ts = frames[0][0]
        last_ts = frames[-1][0]
        count = len(frames)
        return f"You said: {transcript_text}. Processed {count} frames ({first_ts} to {last_ts})."


def tts_synthesize(text: str, out_mp3_path: str) -> None:
    os.makedirs(os.path.dirname(out_mp3_path), exist_ok=True)
    tts = gTTS(text=text, lang="en")
    tts.save(out_mp3_path)