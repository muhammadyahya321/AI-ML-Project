"""Render-ready backend for the RACE Quiz AI inference API."""

from __future__ import annotations

import os
import sys
from functools import lru_cache

from flask import Flask, jsonify, request

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference import RaceInferenceEngine  # noqa: E402

app = Flask(__name__)


@lru_cache(maxsize=1)
def get_engine() -> RaceInferenceEngine:
    """Load inference artifacts once per backend container."""
    return RaceInferenceEngine(model_dir=os.path.join(ROOT_DIR, "models"))


@app.after_request
def add_cors_headers(response):  # type: ignore[no-untyped-def]
    """Allow browser access from a separately hosted frontend."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/api/health", methods=["GET", "OPTIONS"])
def health() -> tuple[object, int]:
    """Return service health and whether trained mode is active."""
    if request.method == "OPTIONS":
        return ("", 204)

    engine = get_engine()
    return jsonify({"ok": True, "demo_mode": engine.use_demo_mode}), 200


@app.route("/api/generate", methods=["POST", "OPTIONS"])
def generate() -> tuple[object, int]:
    """Generate a quiz payload from article text."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or request.form.to_dict()
    article = str(payload.get("article", "")).strip()
    if not article:
        return jsonify({"error": "Article text is required."}), 400

    try:
        result = get_engine().run_full_pipeline(article)
    except Exception as exc:  # pragma: no cover - deployment/runtime path
        return jsonify({"error": str(exc)}), 500

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)
