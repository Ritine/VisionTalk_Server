import threading
import os
from typing import List, Dict, Any, Optional

# Global singleton holder
_MODEL_LOCK = threading.Lock()
_MODEL = None  # type: Optional[object]
_PROCESSOR = None  # type: Optional[object]
_RUNTIME_CFG = {
    "device": "auto",  # auto/cpu/cuda:mps
    "dtype": "auto",
    # Quantization: only applicable on CUDA with bitsandbytes. Set to "none" by default.
    "quant": "none",   # none | bnb4 | bnb8 (CUDA only)
    # Use 3B by default
    "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    # Processor pixel caps to avoid huge buffers
    "min_pixels": 256 * 28 * 28,
    "max_pixels": 1280 * 28 * 28,
}


def load_model_once(**overrides) -> None:
    """Lazily initialize the model once per process.

    Loads Qwen-2.5-VL-3B model and processor for multimodal inference.
    """
    global _MODEL, _PROCESSOR
    with _MODEL_LOCK:
        if _MODEL is not None and _PROCESSOR is not None:
            return

        # Merge overrides to runtime config
        _RUNTIME_CFG.update(overrides)

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            # from transformers import BitsAndBytesConfig

            print(f"Loading model from {_RUNTIME_CFG['model_id']}...")

            # Load the model on the available device(s)
            _MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                _RUNTIME_CFG["model_id"],
                torch_dtype="auto",
                device_map="auto",
            )

            # Load the processor with pixel caps
            _PROCESSOR = AutoProcessor.from_pretrained(
                _RUNTIME_CFG["model_id"],
                min_pixels=_RUNTIME_CFG.get("min_pixels"),
                max_pixels=_RUNTIME_CFG.get("max_pixels"),
            )

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            _MODEL = None
            _PROCESSOR = None
            raise


def generate(messages: List[Dict[str, Any]], max_new_tokens: int = 256, **gen_kwargs) -> str:
    """Run inference with Qwen-2.5-VL-3B on provided messages."""
    if _MODEL is None or _PROCESSOR is None:
        try:
            load_model_once()
        except Exception as e:
            # If model loading fails, return stub response
            image_count = 0
            text_segments = []
            for msg in messages:
                for item in msg.get("content", []):
                    if item.get("type") == "image":
                        image_count += 1
                    elif item.get("type") == "text":
                        text_segments.append(item.get("text", ""))
            transcript_preview = " ".join(text_segments)[:80]
            return f"[Qwen-Stub] images={image_count}; prompt=\"{transcript_preview}...\""

    if _MODEL is None or _PROCESSOR is None:
        image_count = 0
        text_segments = []
        for msg in messages:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    image_count += 1
                elif item.get("type") == "text":
                    text_segments.append(item.get("text", ""))
        transcript_preview = " ".join(text_segments)[:80]
        return f"[Qwen-Stub] images={image_count}; prompt=\"{transcript_preview}...\""

    try:
        from qwen_vl_utils import process_vision_info

        text = _PROCESSOR.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = _PROCESSOR(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(_MODEL.parameters()).device
        inputs = inputs.to(device)
        generated_ids = _MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = _PROCESSOR.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0] if output_text else "No output generated"

    except Exception as e:
        print(f"Error during model inference: {e}")
        image_count = 0
        text_segments = []
        for msg in messages:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    image_count += 1
                elif item.get("type") == "text":
                    text_segments.append(item.get("text", ""))
        transcript_preview = " ".join(text_segments)[:80]
        return f"[Qwen-Error] images={image_count}; prompt=\"{transcript_preview}...\"; error: {str(e)}"