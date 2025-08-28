# RayNeo X2 VisionTalk Server

This server-side application receives audio and image frames from smart glasses clients, orchestrates a minimal viable multimodal pipeline (ASR → Model Inference → TTS), and returns a directly playable audio URL.

## Environment Setup & Installation
- Using system Python (no virtual environment):
```bash
pip install -r requirements.txt
python app.py
```
- Adjust `BASE_URL` in `app.py` as needed: when both glasses and computer connect to phone hotspot, use http://172.20.10.4:5050; when both connect to home Wi-Fi (4THU_6RZZNT), use 192.168.55.114

## Directory Structure
```
VisionTalk_Server/
  app.py                # Flask service, endpoints, indexing & cleanup
  pipeline.py           # ASR/multimodal/TTS placeholder implementations & Qwen input preparation
  qwen_runtime.py       # Qwen-2.5-VL-3B runtime: loading & generate(messages)
  requirements.txt      # Required dependencies
  static/
    test.mp3            # Test audio file
    outputs/            # Generated mp3 outputs, HTTP accessible
  data/
    frames/             # Store frame images: frame_<timestamp>.jpg
    audios/             # Store audio files: audio_<timestamp>.m4a
```

## API Endpoints (Base URL: `http://<server-ip>:5050`)

### POST `/process_frame`
- `multipart/form-data`
- Fields:
  - `image`: Single JPEG (field name must be `image`)
  - `timestamp`: Millisecond timestamp (string)
  - `frame_index`: Optional
- Returns: `{"status":"ok"}`
- Storage: `./data/frames/<timestamp>.jpg`
- Memory index: Maintains ascending frame list by session (MVP uses `default`)

### POST `/process_audio`
- `multipart/form-data`
- Fields:
  - `audio`: m4a file, recommended filename `audio_<startTimestampMs>.m4a`
  - Optional `start_ts`: Read start time from this field when filename cannot be parsed
- Process (frame selection strategy updated):
  1) Parse `start_ts`
  2) Get all frames with `timestamp >= start_ts` from memory index
  3) Apply "uniform sampling" to these frames, keeping max 3 (use all if less than 3)
  4) Run `ASR → multimodal → TTS` pipeline, generate output audio to `./static/outputs/<reqId>.mp3`
- Returns:
  ```json
  {"audio_url":"http://<server-ip>:5050/static/outputs/<id>.mp3","text":"<optional_text>"}
  ```

### POST `/process` **New Endpoint**
- `multipart/form-data`
- Fields:
  - `audio`: Audio file (required)
  - `image`: Image file (required)
- Function: Process a single audio file and image file, run complete pipeline directly
- Returns:
  ```json
  {
    "audio_url": "http://<server-ip>:5050/static/outputs/<id>.mp3",
    "text": "Generated text content",
    "timings_ms": {
      "asr": 123.4,
      "multimodal": 456.7,
      "tts": 89.0,
      "total": 678.9
    }
  }
  ```

### GET `/static/outputs/<id>.mp3`
- Directly returns generated audio for client playback

## Naming & Metadata
- **Images**: Client upload filename `frame_{timestamp_ms}_{frameIndex}.jpg`, server stores as `frame_{timestamp_ms}.jpg`
- **Audio**: Client upload filename `audio_{timestamp_ms}.m4a`, server preserves original filename
- **Output Audio**: Server generates `{reqId}.mp3` to `static/outputs/` directory

## Timeout & Limitations
- Client network constraints: 30s connection, 60s read/write. Ensure `/process_audio` and `/process` complete within 60s
- Frame selection: No fixed time window; only keep frames after `start_ts`, uniform sampling up to 3 frames (modify `MAX_SAMPLED_FRAMES` in `app.py`)

## Pipeline Implementation (Minimal Viable Demo)
- `pipeline.py` currently uses placeholder implementations (stubs):
  - `asr_transcribe(audio_path) -> str`: Returns placeholder transcription text
  - `multimodal_reason(transcript, frames) -> str`:
    - First calls `prepare_qwen_vl_inputs(transcript, frames)` to generate multimodal messages
    - Then calls `qwen_runtime.generate(messages)` to get response
    - Returns placeholder text as fallback if inference fails
  - `tts_synthesize(text, out_mp3_path) -> None`: Copies `static/output_audio.mp3` as synthesis result
- These function signatures are **contracts**. When replacing with real implementations, maintain function names/return values unchanged, no need to modify `app.py`.

## Qwen-2.5-VL-3B Integration
- Input preparation: `prepare_qwen_vl_inputs(transcript, frames, max_frames=30) -> List[dict]`
- Inference runtime: `qwen_runtime.py`
  - `load_model_once(**overrides)`: Lazy load model (not loaded at startup, loaded on first call)
  - `generate(messages: List[dict], max_new_tokens=256, **kwargs) -> str`: Returns response string
  - You only need to replace `load_model_once` and `generate` placeholder implementations with real quantized Qwen-2.5-VL-3B loading and inference (e.g., Transformers/vLLM/LMDeploy, etc.)

- Messages example:
```python
[
  {"role":"system","content":[{"type":"text","text":"You are a helpful multimodal assistant."}]},
  {"role":"user","content":[
     {"type":"image","image":"/abs/path/frames/frame_1712345678901.jpg"},
     ... up to 30 frames ...,
     {"type":"text","text":"ASR transcript...\nPlease answer briefly based on the images (Chinese preferred)."}
  ]}
]
```

## Client-Server Interaction Flow
1. **Traditional Flow**:
   - Client starts speaking: begins recording; takes photos every 2s during recording and calls `/process_frame`
   - Recording ends: uploads entire audio to `/process_audio` and waits for response
   - Server returns `audio_url`: client immediately fetches and plays

2. **Simplified Flow** (Recommended):
   - After client completes recording + photo capture, directly calls `/process` endpoint
   - Server processes everything at once and returns result

## Cleanup Strategy
- Background thread cleans up every 60s, deleting frames, audio, and outputs older than 30 minutes, and synchronously trims memory index
- Parameters configurable at top of `app.py` (`RETENTION_SECONDS`, etc.)

## Security
- Uses `secure_filename` to handle client-provided filenames, preventing directory traversal and abnormal characters

## Debug Examples (curl)

### Send a frame:
```bash
curl -X POST http://<server-ip>:5050/process_frame \
  -F image=@static/input_image.jpg \
  -F timestamp=$(python -c 'import time;print(int(time.time()*1000))')
```

### Send audio:
```bash
# Assuming filename follows audio_<start_ts>.m4a pattern
curl -X POST http://<server-ip>:5050/process_audio \
  -F audio=@static/input_audio.m4a
```

### Use simplified endpoint (Recommended):
```bash
curl -X POST http://<server-ip>:5050/process \
  -F audio=@static/input_audio.m4a \
  -F image=@static/input_image.jpg
```

## Guide for Replacing with Real Implementations
- ASR: Integrate Google Cloud Speech-to-Text or local ASR
- Multimodal: Use Qwen-2.5-VL-3B, `multimodal_reason` internally already connects to `qwen_runtime.generate`
- TTS: Integrate Google Cloud TTS, edge-tts, gTTS, or local TTS (output mp3 to `static/outputs/`)

## Notes
- Demo phase can directly use current placeholder implementations to test client interaction; for production deployment, consider:
  - Refined session management;
  - Authentication/logging/metrics;
  - Storage and concurrency optimization;
  - Choose between in-process or independent inference service based on model size.
- **New `/process` endpoint** simplifies client calling flow, recommended for priority use.