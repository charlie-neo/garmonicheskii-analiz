# api_server.py
import os
import uuid
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify

from analyzer import MusicAnalyzer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

print("=" * 60)
print("🚀 Запуск API-сервера анализа гармонии")
print("=" * 60)
print("📦 Загружаю нейросеть...")
analyzer = MusicAnalyzer(models_dir="dataset/models")
print("✅ Готов к приёму запросов на http://0.0.0.0:5001")
print("=" * 60)

ALLOWED_EXT = {'.mxl', '.xml', '.mid', '.midi'}


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не передан'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Пустое имя файла'}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({
            'error': f'Неверный формат. Поддерживаются: {", ".join(sorted(ALLOWED_EXT))}'
        }), 400

    tmp_dir = Path(tempfile.gettempdir()) / "music_analyzer_api"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}{ext}"

    try:
        f.save(str(tmp_path))
        result = analyzer.analyze(str(tmp_path))

        if result.get('status') == 'error':
            return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': f'Ошибка на сервере: {e}'}), 500

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
