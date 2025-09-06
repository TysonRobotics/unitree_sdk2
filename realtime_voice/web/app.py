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


# Soundboard config helpers
def soundboard_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "soundboard.json"))


def load_soundboard() -> dict:
    try:
        path = soundboard_path()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                if isinstance(cfg, dict) and 'profiles' in cfg:
                    return cfg
        # default config
        return {"slot_count": 16, "profiles": {"Default": []}}
    except Exception:
        return {"slot_count": 16, "profiles": {"Default": []}}


def save_soundboard(cfg: dict) -> bool:
    try:
        path = soundboard_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


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


@app.route("/api/soundboard", methods=["GET"])
def api_soundboard_get():
    cfg = load_soundboard()
    return jsonify({"ok": True, "config": cfg, "recordings": list_recordings()})


@app.route("/api/soundboard/profile", methods=["POST"])
def api_soundboard_profile_create():
    body = request.get_json(silent=True) or {}
    profile = str(body.get("profile") or "").strip()
    try:
        slot_count = int(body.get("slot_count") or 16)
    except Exception:
        slot_count = 16
    if not profile:
        return jsonify({"ok": False, "error": "missing profile"}), 400
    cfg = load_soundboard()
    if "profiles" not in cfg or not isinstance(cfg.get("profiles"), dict):
        cfg["profiles"] = {}
    cfg.setdefault("slot_count", slot_count)
    # Initialize with empty slots
    cfg["profiles"][profile] = [None] * cfg["slot_count"]
    if not save_soundboard(cfg):
        return jsonify({"ok": False, "error": "failed to save"}), 500
    return jsonify({"ok": True, "config": cfg})


@app.route("/api/soundboard/save", methods=["POST"])
def api_soundboard_save():
    body = request.get_json(silent=True) or {}
    profile = str(body.get("profile") or "").strip() or "Default"
    slots = body.get("slots") or []
    if not isinstance(slots, list):
        return jsonify({"ok": False, "error": "slots must be a list"}), 400
    # sanitize filenames to basenames
    clean_slots = []
    for x in slots:
        if not x:
            clean_slots.append(None)
        else:
            name = os.path.basename(str(x))
            clean_slots.append(name)
    cfg = load_soundboard()
    cfg.setdefault("slot_count", 16)
    cfg.setdefault("profiles", {})
    cfg["profiles"][profile] = clean_slots
    if not save_soundboard(cfg):
        return jsonify({"ok": False, "error": "failed to save"}), 500
    return jsonify({"ok": True})


@app.route("/api/soundboard/play", methods=["POST"])
def api_soundboard_play():
    body = request.get_json(silent=True) or {}
    profile = str(body.get("profile") or "").strip() or "Default"
    try:
        index = int(body.get("index", 0))
    except Exception:
        index = 0
    cfg = load_soundboard()
    profiles = cfg.get("profiles", {})
    slots = profiles.get(profile, [])
    if index < 0 or index >= len(slots):
        return jsonify({"ok": False, "error": "index out of range"}), 400
    file_name = slots[index]
    if not file_name:
        return jsonify({"ok": False, "error": "empty slot"}), 400
    # Play via FIFO command
    ok = send_cmd({"cmd": "play", "file": file_name})
    return jsonify({"ok": ok})





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



