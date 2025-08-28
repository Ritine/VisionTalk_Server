## RayNeo X2 VisionTalk Server

本服务端用于接收智能眼镜客户端上传的音频与图片帧，编排最小可行的多模态流水线（ASR → 模型推理 → TTS），并返回可直接播放的音频 URL。

### 运行环境与安装
- 使用系统 Python（不启用虚拟环境）：
```bash
pip install -r requirements.txt
python app.py
```
- 按需在 `app.py` 中调整 `BASE_URL`，如果眼镜和电脑都连手机热点时电脑的IP地址是http://172.20.10.4:5050，眼镜和电脑都连家里WI-FI（4THU_6RZZNT）时电脑的IP地址是192.168.55.114

### 目录结构
```
VisionTalk_Server/
  app.py                # Flask 服务、端点、索引与清理
  pipeline.py           # ASR/多模态/ TTS 的占位实现与 Qwen 输入准备
  qwen_runtime.py       # Qwen-2.5-VL-3B 运行时：加载与 generate(messages)
  requirements.txt      # 必需依赖
  static/
    test.mp3            # 测试用音频
    outputs/            # 合成的 mp3 输出，HTTP 可访问
  data/
    frames/             # 存储帧图像：frame_<timestamp>.jpg
    audios/             # 存储音频：audio_<timestamp>.m4a
```

### 接口约定（基地址：`http://<server-ip>:5050`）

#### 1. POST `/process_frame`
- `multipart/form-data`
- 字段：
  - `image`: JPEG 单张（字段名固定 `image`）
  - `timestamp`: 毫秒时间戳（字符串）
  - `frame_index`: 可选
- 返回：`{"status":"ok"}`
- 存储：`./data/frames/<timestamp>.jpg`
- 内存索引：按会话（MVP 为 `default`）维护升序帧列表

#### 2. POST `/process_audio`
- `multipart/form-data`
- 字段：
  - `audio`: m4a 文件，文件名推荐 `audio_<startTimestampMs>.m4a`
  - 可选 `start_ts`：当文件名无法解析时从该字段读取开始时间
- 流程（帧选择策略 已更新）：
  1) 解析 `start_ts`
  2) 从内存索引取所有 `timestamp >= start_ts` 的帧
  3) 对这些帧做"均匀采样"，最多保留 3 张（不足 3 张则全用）
  4) `ASR → multimodal → TTS` 生成输出音频到 `./static/outputs/<reqId>.mp3`
- 返回：
  ```json
  {"audio_url":"http://<server-ip>:5050/static/outputs/<id>.mp3","text":"<optional_text>"}
  ```

#### 3. POST `/process` ⭐ **新增接口**
- `multipart/form-data`
- 字段：
  - `audio`: 音频文件（必需）
  - `image`: 图像文件（必需）
- 功能：单次处理一个音频文件和一个图像文件，直接运行完整流水线
- 返回：
  ```json
  {
    "audio_url": "http://<server-ip>:5050/static/outputs/<id>.mp3",
    "text": "生成的文本内容",
    "timings_ms": {
      "asr": 123.4,
      "multimodal": 456.7,
      "tts": 89.0,
      "total": 678.9
    }
  }
  ```

#### 4. GET `/static/outputs/<id>.mp3`
- 直接返回合成音频，供客户端播放

### 命名与元数据
- **图片**：客户端上传文件名 `frame_{timestamp_ms}_{frameIndex}.jpg`，服务端存储为 `frame_{timestamp_ms}.jpg`
- **音频**：客户端上传文件名 `audio_{timestamp_ms}.m4a`，服务端保持原文件名
- **输出音频**：服务端生成 `{reqId}.mp3` 到 `static/outputs/` 目录

### 超时与限制
- 客户端网络约束：连接 30s，读/写 60s。请确保 `/process_audio` 和 `/process` 在 60s 内完成
- 帧选择：不使用固定时间窗；仅保留 `start_ts` 之后的帧，均匀采样至多 3 张（可在 `app.py` 修改 `MAX_SAMPLED_FRAMES`）

### 流水线实现（最小可行 Demo）
- `pipeline.py` 当前使用占位实现（stubs）：
  - `asr_transcribe(audio_path) -> str`：返回占位转写文本
  - `multimodal_reason(transcript, frames) -> str`：
    - 先调用 `prepare_qwen_vl_inputs(transcript, frames)` 生成多模态 messages
    - 再调用 `qwen_runtime.generate(messages)` 获取回答
    - 若推理报错，返回占位文本兜底
  - `tts_synthesize(text, out_mp3_path) -> None`：复制 `static/output_audio.mp3` 作为合成结果
- 这些函数签名是 **契约**。替换为真实实现时保持函数名/返回值不变，无需改 `app.py`。

### Qwen-2.5-VL-3B 接入
- 输入准备：`prepare_qwen_vl_inputs(transcript, frames, max_frames=30) -> List[dict]`
- 推理运行时：`qwen_runtime.py`
  - `load_model_once(**overrides)`: 懒加载模型（启动时不加载，首次调用时加载）
  - `generate(messages: List[dict], max_new_tokens=256, **kwargs) -> str`: 返回回答字符串
  - 你只需把 `load_model_once` 与 `generate` 的占位实现替换为真实的量化 Qwen-2.5-VL-3B 加载与推理（如 Transformers/vLLM/LMDeploy 等）

- messages 示例：
```python
[
  {"role":"system","content":[{"type":"text","text":"You are a helpful multimodal assistant."}]},
  {"role":"user","content":[
     {"type":"image","image":"/abs/path/frames/frame_1712345678901.jpg"},
     ... 最多 30 张帧 ...,
     {"type":"text","text":"ASR transcript...\n请结合图像简要回答（中文优先）。"}
  ]}
]
```

### 与客户端的交互流程
1. **传统流程**：
   - 客户端开始说话：启动录音；录音期间每 2s 拍照并调用 `/process_frame`
   - 录音结束：上传整段音频到 `/process_audio` 并等待响应
   - 服务端返回 `audio_url`：客户端立即拉取并播放

2. **简化流程**（推荐）：
   - 客户端录音+拍照完成后，直接调用 `/process` 接口
   - 服务端一次性处理并返回结果

### 清理策略
- 后台线程每 60s 清理一次，删除 30 分钟前的帧、音频与输出，并同步修剪内存索引
- 参数可在 `app.py` 顶部配置（`RETENTION_SECONDS` 等）

### 安全性
- 使用 `secure_filename` 处理客户端提供的文件名，避免目录穿越与异常字符

### 调试示例（curl）

#### 发送一张帧：
```bash
curl -X POST http://<server-ip>:5050/process_frame \
  -F image=@static/input_image.jpg \
  -F timestamp=$(python -c 'import time;print(int(time.time()*1000))')
```

#### 发送音频：
```bash
# 假设文件名符合 audio_<start_ts>.m4a
curl -X POST http://<server-ip>:5050/process_audio \
  -F audio=@static/input_audio.m4a
```

#### 使用简化接口（推荐）：
```bash
curl -X POST http://<server-ip>:5050/process \
  -F audio=@static/input_audio.m4a \
  -F image=@static/input_image.jpg
```

### 替换为真实实现的指南
- ASR：接入 Google Cloud Speech-to-Text 或本地 ASR
- 多模态：使用Qwen-2.5-VL-3B，`multimodal_reason` 内部已对接 `qwen_runtime.generate`
- TTS：接入 Google Cloud TTS、edge-tts、gTTS 或本地 TTS（输出 mp3 到 `static/outputs/`）

### 备注
- Demo 阶段可直接使用当前占位实现跑通客户端交互；上线时建议：
  - 细化 session 管理；
  - 增加鉴权/日志/指标；
  - 优化存储与并发；
  - 根据模型大小选择进程内或独立推理服务方案。
- **新增的 `/process` 接口** 简化了客户端调用流程，推荐优先使用。