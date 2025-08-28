from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/process', methods=['POST'])
def process():
    audio = request.files.get('audio')
    image = request.files.get('image')

    if audio:
        filename = secure_filename(audio.filename)  # 保留原始文件名（做安全处理）
        audio.save(os.path.join(UPLOAD_FOLDER, filename))
    if image:
        filename = secure_filename(image.filename)
        image.save(os.path.join(UPLOAD_FOLDER, filename))

    # 假装处理完，返回一个固定音频地址
    return jsonify({
        "audio_url": "http://192.168.55.114:5050/static/output_audio.mp3"  # 改成你实际地址
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)