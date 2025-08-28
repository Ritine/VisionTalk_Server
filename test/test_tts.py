import os, time
from gtts import gTTS

def tts_synthesize(text: str, out_mp3_path: str) -> None:
    os.makedirs(os.path.dirname(out_mp3_path), exist_ok=True)
    tts = gTTS(text=text)
    tts.save(out_mp3_path)

STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")
# req_id = str(int(time.time() * 1000))
out_mp3_path = os.path.join(OUTPUTS_DIR, f"test.mp3")

tts_synthesize("What kind of pizza is this? Is it suitable for a vegetarian?", out_mp3_path)