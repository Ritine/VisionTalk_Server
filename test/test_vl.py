import argparse
import os
from typing import List, Dict, Any

from qwen_runtime import generate, load_model_once

# # 单图
# python test_vl.py static/input_image.jpg --text "用中文概括图片中的内容"

# # 多图
# python test_vl.py data/frames/1712345678901.jpg data/frames/1712345680000.jpg --text "结合帧描述发生了什么"


def build_messages(image_paths: List[str], text: str) -> List[Dict[str, Any]]:
    content = []
    for p in image_paths:
        abs_path = os.path.abspath(p)
        content.append({"type": "image", "image": abs_path})
    content.append({"type": "text", "text": text})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful multimodal assistant."}]},
        {"role": "user", "content": content},
    ]
    return messages


def main():
    parser = argparse.ArgumentParser(description="Test Qwen-2.5-VL-3B inference")
    parser.add_argument("images", nargs="+", help="One or more image paths")
    parser.add_argument("--text", required=True, help="User prompt / ASR transcript")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    # Validate images
    for p in args.images:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")

    # Optional: preload to surface errors early
    try:
        load_model_once()
    except Exception as e:
        print(f"[Warn] Model preload failed, will fallback to stub at runtime: {e}")

    messages = build_messages(args.images, args.text)
    output = generate(messages, max_new_tokens=args.max_new_tokens)
    print("=== Qwen VL Output ===")
    print(output)


if __name__ == "__main__":
    main()
