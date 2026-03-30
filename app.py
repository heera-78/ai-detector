# app.py

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import torch

app = Flask(__name__)
CORS(app)

print("Loading cyberbullying detection model...")

classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-offensive",
    return_all_scores=True
)

print("Model loaded and ready!")

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

        results = classifier(text)

        # Flatten if nested list
        if isinstance(results[0], list):
            results = results[0]

        print("Raw results:", results)
        print("All labels:", [r["label"] for r in results])

        scores = { r["label"]: r["score"] for r in results }
        print("Scores dict:", scores)

        # Check all possible label names
        offensive_score = (
            scores.get("offensive") or
            scores.get("LABEL_1") or
            scores.get("toxic") or
            0
        )

        non_offensive_score = (
            scores.get("non-offensive") or
            scores.get("LABEL_0") or
            scores.get("non_offensive") or
            0
        )

        print(f"offensive_score: {offensive_score}")
        print(f"non_offensive_score: {non_offensive_score}")

        # If we only got non-offensive score, derive offensive
        if offensive_score == 0 and non_offensive_score > 0:
            offensive_score = 1 - non_offensive_score

        is_bullying = offensive_score > 0.7

        return jsonify({
            "text": text,
            "isBullying": is_bullying,
            "score": round(offensive_score, 4),
            "confidence": f"{round(offensive_score * 100, 1)}%",
            "debug_scores": scores
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False, use_reloader=False)