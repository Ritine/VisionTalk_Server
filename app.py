from flask import Flask, request, jsonify, send_from_directory
import os
import threading
import time
from typing import Dict, List, Tuple
from werkzeug.utils import secure_filename
from pipeline import asr_transcribe, multimodal_reason, tts_synthesize
from qwen_runtime import load_model_once

# Configuration
# Set IP based on network: phone hotspot -> 172.20.10.4, home Wi-Fi (4THU_6RZZNT) -> 192.168.55.114
IP = "192.168.55.114"
# IP = "172.20.10.4"
BASE_URL = f"http://{IP}:5050"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
AUDIOS_DIR = os.path.join(DATA_DIR, "audios")
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")

# Use all frames >= start_ts and uniformly sample up to this count
MAX_SAMPLED_FRAMES = 3

RETENTION_SECONDS = 30 * 60  # 30 minutes

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(AUDIOS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR)

# In-memory index: session -> List[Tuple[int, str]] sorted by timestamp
frames_index_lock = threading.Lock()
frames_index: Dict[str, List[Tuple[int, str]]] = {}


def get_session_id() -> str:
    # MVP: single session. Extend to parse from headers/form if needed.
    return "default"


def evenly_sample(items: List[Tuple[int, str]], k: int) -> List[Tuple[int, str]]:
    if k <= 0:
        return []
    n = len(items)
    if n <= k:
        return items
    # Evenly spaced indices including first and last
    indices = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    # Deduplicate while preserving order (rounding could duplicate)
    seen = set()
    sampled = []
    for idx in indices:
        if idx not in seen:
            sampled.append(items[idx])
            seen.add(idx)
    # If dedup reduced count, backfill by linear scan
    i = 0
    while len(sampled) < k and i < n:
        if i not in seen:
            sampled.append(items[i])
        i += 1
    # Preserve chronological order
    sampled.sort(key=lambda x: x[0])
    return sampled


@app.route("/process_frame", methods=["POST"])
def process_frame():
    print("[process_frame] request received")
    if "image" not in request.files:
        print("[process_frame] missing image")
        return jsonify({"error": "missing image"}), 400

    image_file = request.files["image"]
    timestamp_str = request.form.get("timestamp")
    frame_index = request.form.get("frame_index")  # optional, not used in MVP

    if not timestamp_str or not timestamp_str.isdigit():
        print(f"[process_frame] invalid timestamp: {timestamp_str}")
        return jsonify({"error": "invalid or missing timestamp"}), 400

    timestamp_ms = int(timestamp_str)

    # Save image as ./data/frames/<timestamp>.jpg
    filename = f"{timestamp_ms}.jpg"
    filename = secure_filename(filename)
    save_path = os.path.join(FRAMES_DIR, filename)
    image_file.save(save_path)
    print(f"[process_frame] saved image -> {save_path}")

    # Update in-memory index
    session_id = get_session_id()
    with frames_index_lock:
        entries = frames_index.setdefault(session_id, [])
        entries.append((timestamp_ms, save_path))
        entries.sort(key=lambda x: x[0])
        print(f"[process_frame] index size (session={session_id}) -> {len(entries)}")

    return jsonify({"status": "ok"})


@app.route("/process_audio", methods=["POST"])
def process_audio():
    t_total_start = time.time()
    print("[process_audio] request received")
    if "audio" not in request.files:
        print("[process_audio] missing audio")
        return jsonify({"error": "missing audio"}), 400

    audio_file = request.files["audio"]

    # Expected filename: audio_<startTimestampMs>.m4a
    original_name = secure_filename(audio_file.filename or "")
    start_ts_ms = None
    if original_name.startswith("audio_") and "." in original_name:
        try:
            start_part = original_name.split("_")[1].split(".")[0]
            start_ts_ms = int(start_part)
        except Exception:
            start_ts_ms = None

    if start_ts_ms is None:
        # Fallback to form field
        start_ts_field = request.form.get("start_ts")
        if start_ts_field and start_ts_field.isdigit():
            start_ts_ms = int(start_ts_field)
        else:
            print("[process_audio] cannot parse start timestamp")
            return jsonify({"error": "cannot parse start timestamp from filename or form"}), 400

    # Persist audio
    audio_filename = f"audio_{start_ts_ms}.m4a"
    audio_save_path = os.path.join(AUDIOS_DIR, audio_filename)
    audio_file.save(audio_save_path)
    print(f"[process_audio] saved audio -> {audio_save_path}")

    # Collect frames: all timestamps >= start_ts
    session_id = get_session_id()
    with frames_index_lock:
        session_frames = frames_index.get(session_id, [])
        candidate_frames = [(ts, path) for (ts, path) in session_frames if ts >= start_ts_ms]
    print(f"[process_audio] candidate frames >= {start_ts_ms} -> {len(candidate_frames)}")

    selected_frames = evenly_sample(candidate_frames, MAX_SAMPLED_FRAMES)
    print(f"[process_audio] sampled frames -> {len(selected_frames)}")

    # Pipeline
    try:
        t_asr_start = time.time()
        print("[process_audio] ASR start")
        transcript = asr_transcribe(audio_save_path)
        t_asr = (time.time() - t_asr_start) * 1000
        print(f"[process_audio] ASR done in {t_asr:.1f} ms, text preview: {str(transcript)[:60]}")

        t_mm_start = time.time()
        print("[process_audio] Multimodal generation start")
        output_text = multimodal_reason(transcript, selected_frames)
        t_mm = (time.time() - t_mm_start) * 1000
        print(f"[process_audio] Multimodal done in {t_mm:.1f} ms, text preview: {str(output_text)[:60]}")

        req_id = str(int(time.time() * 1000))
        out_mp3_path = os.path.join(OUTPUTS_DIR, f"{req_id}.mp3")
        t_tts_start = time.time()
        print("[process_audio] TTS start")
        tts_synthesize(output_text, out_mp3_path)
        t_tts = (time.time() - t_tts_start) * 1000
        print(f"[process_audio] TTS done in {t_tts:.1f} ms -> {out_mp3_path}")

        audio_url = f"{BASE_URL}/static/outputs/{req_id}.mp3"
        t_total = (time.time() - t_total_start) * 1000
        print(f"[process_audio] returning audio_url -> {audio_url}; total {t_total:.1f} ms")
        return jsonify({"audio_url": audio_url, "text": output_text, "timings_ms": {"asr": t_asr, "multimodal": t_mm, "tts": t_tts, "total": t_total}})
    except Exception as e:
        print(f"[process_audio] error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/process", methods=["POST"])
def process_single_audio_image():
    """Accept a single audio and a single image, run the pipeline, and return audio_url + text.
    Expected form fields: 'audio' (file), 'image' (file). Others optional.
    """
    t_total_start = time.time()
    print("[process] request received")
    audio_file = request.files.get("audio")
    image_file = request.files.get("image")

    if not audio_file and not image_file:
        print("[process] missing audio and image")
        return jsonify({"error": "missing audio and image"}), 400
    if not audio_file:
        print("[process] missing audio")
        return jsonify({"error": "missing audio"}), 400
    if not image_file:
        print("[process] missing image")
        return jsonify({"error": "missing image"}), 400

    # Save audio
    now_ms = int(time.time() * 1000)
    audio_name = secure_filename(audio_file.filename or f"audio_{now_ms}.m4a")
    audio_save_path = os.path.join(AUDIOS_DIR, audio_name)
    audio_file.save(audio_save_path)
    print(f"[process] saved audio -> {audio_save_path}")

    # Save image; use current timestamp as its logical timestamp
    image_name = secure_filename(image_file.filename or f"{now_ms}.jpg")
    image_save_path = os.path.join(FRAMES_DIR, image_name)
    image_file.save(image_save_path)
    print(f"[process] saved image -> {image_save_path}")

    # Build a single-frame list with this image. Timestamp extracted if name starts with digits, otherwise use now.
    ts_for_frame = now_ms
    try:
        leading = os.path.splitext(image_name)[0]
        if leading.isdigit():
            ts_for_frame = int(leading)
    except Exception:
        pass

    selected_frames: List[Tuple[int, str]] = [(ts_for_frame, image_save_path)]

    # Pipeline
    try:
        t_asr_start = time.time()
        print("[process] ASR start")
        transcript = asr_transcribe(audio_save_path)
        t_asr = (time.time() - t_asr_start) * 1000
        print(f"[process] ASR done in {t_asr:.1f} ms, text preview: {str(transcript)[:60]}")

        t_mm_start = time.time()
        print("[process] Multimodal generation start")
        output_text = multimodal_reason(transcript, selected_frames)
        t_mm = (time.time() - t_mm_start) * 1000
        print(f"[process] Multimodal done in {t_mm:.1f} ms, text preview: {str(output_text)[:60]}")

        req_id = str(int(time.time() * 1000))
        out_mp3_path = os.path.join(OUTPUTS_DIR, f"{req_id}.mp3")
        t_tts_start = time.time()
        print("[process] TTS start")
        tts_synthesize(output_text, out_mp3_path)
        t_tts = (time.time() - t_tts_start) * 1000
        print(f"[process] TTS done in {t_tts:.1f} ms -> {out_mp3_path}")

        audio_url = f"{BASE_URL}/static/outputs/{req_id}.mp3"
        t_total = (time.time() - t_total_start) * 1000
        print(f"[process] returning audio_url -> {audio_url}; total {t_total:.1f} ms")
        return jsonify({"audio_url": audio_url, "text": output_text, "timings_ms": {"asr": t_asr, "multimodal": t_mm, "tts": t_tts, "total": t_total}})
    except Exception as e:
        print(f"[process] error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/static/outputs/<path:filename>', methods=['GET'])
def serve_output(filename: str):
    return send_from_directory(OUTPUTS_DIR, filename, as_attachment=False)


# Background cleanup

def _delete_older_than(dir_path: str, cutoff_epoch_s: float):
    try:
        for name in os.listdir(dir_path):
            try:
                fpath = os.path.join(dir_path, name)
                if not os.path.isfile(fpath):
                    continue
                mtime = os.path.getmtime(fpath)
                if mtime < cutoff_epoch_s:
                    os.remove(fpath)
            except Exception:
                continue
    except FileNotFoundError:
        return


def cleanup_loop():
    while True:
        cutoff = time.time() - RETENTION_SECONDS
        _delete_older_than(FRAMES_DIR, cutoff)
        _delete_older_than(AUDIOS_DIR, cutoff)
        _delete_older_than(OUTPUTS_DIR, cutoff)

        # Prune frames_index entries not on disk or too old
        with frames_index_lock:
            for session_id, entries in list(frames_index.items()):
                pruned: List[Tuple[int, str]] = []
                for ts, path in entries:
                    if (ts / 1000.0) < cutoff:
                        continue
                    if os.path.exists(path):
                        pruned.append((ts, path))
                frames_index[session_id] = pruned
        time.sleep(60)


cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
cleanup_thread.start()


if __name__ == "__main__":
    # Preload model at startup to avoid first request delay
    print("Preloading Qwen-2.5-VL-3B model...")
    try:
        load_model_once()
        print("Model preloaded successfully!")
    except Exception as e:
        print(f"Warning: Model preloading failed: {e}")
        print("Model will be loaded lazily on first request (may cause delay)")
    
    # Host 0.0.0.0 to be reachable from RayNeo on same network
    app.run(host="0.0.0.0", port=5050, debug=False)