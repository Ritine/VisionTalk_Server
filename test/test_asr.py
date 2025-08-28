import sys
import os
from pipeline import asr_transcribe


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_asr.py <audio_path> [lang]")
        print("Example: python test_asr.py static/input_audio.m4a zh-CN")
        sys.exit(1)

    audio_path = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) >= 3 else "zh-CN"

    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(2)

    print(f"[ASR] Transcribing: {audio_path} (lang={lang})")
    text = asr_transcribe(audio_path, language=lang)
    print("[ASR] Result:")
    print(text)


if __name__ == "__main__":
    main()
