import json
import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from typing import List

APP_PORT = int(os.environ.get("VOICE_WEB_PORT", "9000"))
FIFO_PATH = os.environ.get(
    "VOICE_CONTROL_FIFO",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "control.fifo"),
)

app = Flask(__name__, static_folder="static", template_folder="static")


def send_cmd(cmd: dict) -> bool:
    try:
        line = json.dumps(cmd) + "\n"
        with open(FIFO_PATH, "w") as f:
            f.write(line)
            f.flush()
        return True
    except Exception as exc:
        print(f"[WEB] Failed to send cmd: {exc}")
        return False


def recordings_dir() -> str:
    # Mirror client's default RECORDINGS_DIR
    return os.environ.get(
        "RECORDINGS_DIR",
        os.path.join(os.path.dirname(__file__), os.pardir, "recordings"),
    )


def list_recordings() -> List[str]:
    try:
        root = os.path.abspath(recordings_dir())
        os.makedirs(root, exist_ok=True)
        files = [f for f in os.listdir(root) if f.lower().endswith('.wav')]
        files.sort(reverse=True)
        return files
    except Exception:
        return []


def is_wav_filename(name: str) -> bool:
    return name.lower().endswith('.wav') and len(name) > 4


@app.route("/api/mute", methods=["POST"])
def api_mute():
    body = request.get_json(silent=True) or {}
    val = bool(body.get("value", True))
    ok = send_cmd({"cmd": "mute", "value": val})
    return jsonify({"ok": ok})


@app.route("/api/toggle_mute", methods=["POST"])
def api_toggle_mute():
    ok = send_cmd({"cmd": "toggle_mute"})
    return jsonify({"ok": ok})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    ok = send_cmd({"cmd": "stop"})
    return jsonify({"ok": ok})


@app.route("/api/reload", methods=["POST"])
def api_reload():
    ok = send_cmd({"cmd": "reload"})
    return jsonify({"ok": ok})


@app.route("/api/volume", methods=["POST"])
def api_volume():
    body = request.get_json(silent=True) or {}
    try:
        val = int(body.get("value", 100))
    except Exception:
        val = 100
    ok = send_cmd({"cmd": "volume", "value": int(max(0, min(100, val)))})
    return jsonify({"ok": ok})


@app.route("/api/volume_up", methods=["POST"])
def api_volume_up():
    ok = send_cmd({"cmd": "volume_up"})
    return jsonify({"ok": ok})


@app.route("/api/volume_down", methods=["POST"])
def api_volume_down():
    ok = send_cmd({"cmd": "volume_down"})
    return jsonify({"ok": ok})


@app.route("/api/barge_in", methods=["POST"])
def api_barge_in():
    body = request.get_json(silent=True) or {}
    val = bool(body.get("value", False))
    ok = send_cmd({"cmd": "barge_in", "value": val})
    return jsonify({"ok": ok})


@app.route("/api/recordings", methods=["GET"])
def api_recordings_list():
    return jsonify({"ok": True, "files": list_recordings()})


@app.route("/api/record_start", methods=["POST"])
def api_record_start():
    body = request.get_json(silent=True) or {}
    name = str(body.get("name") or "").strip()
    ok = send_cmd({"cmd": "record_start", "name": name})
    return jsonify({"ok": ok})


@app.route("/api/record_stop", methods=["POST"])
def api_record_stop():
    ok = send_cmd({"cmd": "record_stop"})
    return jsonify({"ok": ok})


@app.route("/api/play", methods=["POST"])
def api_play():
    body = request.get_json(silent=True) or {}
    file_name = str(body.get("file") or "").strip()
    if not file_name:
        return jsonify({"ok": False, "error": "missing file"}), 400
    # Only allow files under recordings dir
    ok = send_cmd({"cmd": "play", "file": file_name})
    return jsonify({"ok": ok})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        if 'file' not in request.files:
            return jsonify({"ok": False, "error": "missing file"}), 400
        file = request.files['file']
        if file.filename is None or file.filename.strip() == '':
            return jsonify({"ok": False, "error": "empty filename"}), 400
        filename = secure_filename(file.filename)
        if not is_wav_filename(filename):
            return jsonify({"ok": False, "error": "only .wav accepted"}), 400
        root = os.path.abspath(recordings_dir())
        os.makedirs(root, exist_ok=True)
        dest = os.path.join(root, filename)
        file.save(dest)
        return jsonify({"ok": True, "file": filename})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500





@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename: str):
    return send_from_directory(app.static_folder, filename)


def main():
    print(f"[WEB] Starting on 0.0.0.0:{APP_PORT}, FIFO at {FIFO_PATH}")
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)


if __name__ == "__main__":
    main()



