# app.py

import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
MODEL_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-offensive"

print("AI Detector ready — using HuggingFace Inference API!")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({ "status": "ok" })

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({ "error": "No text provided" }), 400

        # Call HuggingFace Inference API
        response = requests.post(
            MODEL_URL,
            headers={ "Authorization": f"Bearer {HF_API_TOKEN}" },
            json={ "inputs": text }
        )

        result = response.json()
        print("HF result:", result)

        # Handle model loading (first request may need to wait)
        if isinstance(result, dict) and "error" in result:
            return jsonify({
                "error": result["error"],
                "isBullying": False,
                "score": 0,
                "confidence": "0%"
            }), 200

        # Flatten if nested
        if isinstance(result[0], list):
            result = result[0]

        scores = { r["label"]: r["score"] for r in result }
        print("Scores:", scores)

        offensive_score = scores.get("offensive", 0)
        non_offensive_score = scores.get("non-offensive", 0)

        if offensive_score == 0 and non_offensive_score > 0:
            offensive_score = 1 - non_offensive_score

        is_bullying = offensive_score > 0.7

        return jsonify({
            "text": text,
            "isBullying": is_bullying,
            "score": round(offensive_score, 4),
            "confidence": f"{round(offensive_score * 100, 1)}%"
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)