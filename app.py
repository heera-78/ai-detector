# app.py
import os
import base64
import requests
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# ── Text / comment offensive detection ───────────────────────────────────────
TEXT_MODEL_URL = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-offensive"

# ── Image / Video moderation — TWO models run on every image ─────────────────
# Model 1: AdamCodd/vit-base-nsfw-detector  — stricter, catches suggestive/swimwear/nudity
# Model 2: Falconsai/nsfw_image_detection   — broader net for explicit/offensive content
# Image is BLOCKED if EITHER model exceeds its threshold.
MODEL_1_URL = "https://router.huggingface.co/hf-inference/models/AdamCodd/vit-base-nsfw-detector"
MODEL_2_URL = "https://router.huggingface.co/hf-inference/models/Falconsai/nsfw_image_detection"

MODEL_1_THRESHOLD = 0.40   # 40% — catches suggestive/swimwear/partial nudity
MODEL_2_THRESHOLD = 0.55   # 55% — broader backup

MAX_VIDEO_FRAMES = 8

print("AI Detector ready — /detect + /detect-post + /detect-media (dual-model)!")


# ── Shared helper: call HF text model ────────────────────────────────────────
def _call_text_model(text: str):
    response = requests.post(
        TEXT_MODEL_URL,
        headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
        json={"inputs": text},
        timeout=30,
    )
    result = response.json()

    if isinstance(result, dict) and "error" in result:
        return None, result["error"]

    if isinstance(result[0], list):
        result = result[0]

    scores = {r["label"]: r["score"] for r in result}
    offensive_score = scores.get("offensive", 0)
    non_offensive_score = scores.get("non-offensive", 0)
    if offensive_score == 0 and non_offensive_score > 0:
        offensive_score = 1 - non_offensive_score

    return offensive_score, None


# ── Shared helpers: image/video scanning ─────────────────────────────────────
def _hf_headers():
    return {"Authorization": f"Bearer {HF_API_TOKEN}"}


def _call_hf_image_model(model_url: str, image_bytes: bytes):
    """Call a HF image classification model. Returns scores dict or None on error."""
    try:
        resp = requests.post(
            model_url,
            headers={**_hf_headers(), "Content-Type": "application/octet-stream"},
            data=image_bytes,
            timeout=30,
        )
        result = resp.json()
        print(f"HF [{model_url.split('/')[-1]}]:", result)

        if isinstance(result, dict) and "error" in result:
            print("Model error:", result["error"])
            return None

        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            result = result[0]

        return {r["label"].lower(): r["score"] for r in result}
    except Exception as e:
        print(f"Model call failed ({model_url}):", e)
        return None


def _nsfw_score_from(scores) -> float:
    if scores is None:
        return 0.0
    nsfw = scores.get("nsfw", 0.0)
    normal = scores.get("normal", None)
    if nsfw == 0.0 and normal is not None:
        nsfw = 1.0 - normal
    return nsfw


def _scan_image_bytes(image_bytes: bytes) -> dict:
    """Run both models. Block if EITHER exceeds its threshold."""
    scores1 = _call_hf_image_model(MODEL_1_URL, image_bytes)
    scores2 = _call_hf_image_model(MODEL_2_URL, image_bytes)

    nsfw1 = _nsfw_score_from(scores1)
    nsfw2 = _nsfw_score_from(scores2)

    blocked_by_1 = nsfw1 >= MODEL_1_THRESHOLD
    blocked_by_2 = nsfw2 >= MODEL_2_THRESHOLD
    any_blocked  = blocked_by_1 or blocked_by_2
    worst_score  = max(nsfw1, nsfw2)

    if blocked_by_1 and blocked_by_2:
        trigger = "Nudity, explicit, or highly sensitive content detected."
    elif blocked_by_1:
        trigger = "Sensitive content detected (nudity / suggestive imagery)."
    elif blocked_by_2:
        trigger = "Explicit or offensive content detected."
    else:
        trigger = "Content appears safe."

    reason = f"Upload blocked — {trigger}" if any_blocked else "Content appears safe."

    return {
        "blocked":      any_blocked,
        "label":        "nsfw" if any_blocked else "normal",
        "score":        round(worst_score, 4),
        "confidence":   f"{round(worst_score * 100, 1)}%",
        "reason":       reason,
        "model_scores": {
            "vit_nsfw_detector": round(nsfw1, 4),
            "falconsai":         round(nsfw2, 4),
        },
    }


def _bytes_from_request_file(file_storage) -> bytes:
    return file_storage.read()


def _bytes_from_base64(b64_string: str) -> bytes:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    return base64.b64decode(b64_string)


def _bytes_from_url(url: str) -> bytes:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.content


def _extract_video_frames(video_bytes: bytes, max_frames: int = MAX_VIDEO_FRAMES):
    """Extract evenly-spaced frames from a video using OpenCV."""
    try:
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            os.unlink(tmp_path)
            return []

        step = max(1, total_frames // max_frames)
        frames = []
        frame_idx = 0

        while len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            _, buf = cv2.imencode(".jpg", frame)
            frames.append(buf.tobytes())
            frame_idx += step

        cap.release()
        os.unlink(tmp_path)
        return frames

    except ImportError:
        raise RuntimeError(
            "opencv-python is required for video scanning. "
            "Install it with: pip install opencv-python-headless"
        )


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── /detect — comment / text bullying detection ───────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    """
    Body: { "text": "..." }
    Returns: { text, isBullying, score, confidence }
    """
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        offensive_score, err = _call_text_model(text)

        if err:
            return jsonify({
                "error":      err,
                "isBullying": False,
                "score":      0,
                "confidence": "0%",
            }), 200

        is_bullying = offensive_score > 0.7

        return jsonify({
            "text":       text,
            "isBullying": is_bullying,
            "score":      round(offensive_score, 4),
            "confidence": f"{round(offensive_score * 100, 1)}%",
        })

    except Exception as e:
        print("Error in /detect:", str(e))
        return jsonify({"error": str(e)}), 500


# ── /detect-post — post caption moderation with severity tiers ───────────────
@app.route("/detect-post", methods=["POST"])
def detect_post():
    """
    Body: { "caption": "...", "imageTags": [] }
    Returns: { caption, severity, score, confidence, message }
    severity: "safe" | "warning" | "blocked"
    """
    try:
        data        = request.get_json()
        caption     = data.get("caption", "").strip()
        image_tags  = data.get("imageTags", [])   # optional context tags from image classifier

        if not caption and not image_tags:
            return jsonify({"error": "No caption or image data provided"}), 400

        # Combine caption + any image tags for richer context
        combined_text = caption
        if image_tags:
            combined_text += " " + " ".join(image_tags)

        offensive_score, err = _call_text_model(combined_text)

        if err:
            return jsonify({
                "error":    err,
                "severity": "safe",
                "score":    0,
            }), 200

        # Severity tiers
        if offensive_score > 0.85:
            severity = "blocked"
            message  = "This post contains severely inappropriate content and cannot be published."
        elif offensive_score > 0.5:
            severity = "warning"
            message  = "This post may contain offensive or sensitive content. Are you sure you want to post this?"
        else:
            severity = "safe"
            message  = ""

        return jsonify({
            "caption":    caption,
            "severity":   severity,
            "score":      round(offensive_score, 4),
            "confidence": f"{round(offensive_score * 100, 1)}%",
            "message":    message,
        })

    except Exception as e:
        print("Error in /detect-post:", str(e))
        return jsonify({"error": str(e)}), 500


# ── /detect-media — image & video NSFW/content moderation ────────────────────
@app.route("/detect-media", methods=["POST"])
def detect_media():
    """
    Accepts:
      • multipart/form-data — field "file" (image or video)
      • application/json   — { "base64": "...", "type": "image"|"video" }
                           — { "url": "...",    "type": "image" }

    Returns:
      {
        "blocked":      true | false,
        "mediaType":    "image" | "video",
        "label":        "nsfw" | "normal",
        "score":        0.91,
        "confidence":   "91.0%",
        "reason":       "...",
        "model_scores": { "vit_nsfw_detector": 0.91, "falconsai": 0.78 }
      }
    """
    try:
        media_type       = "image"
        image_bytes_list = []
        is_video         = False

        # ── 1. Multipart upload ──────────────────────────────────────────────
        if "file" in request.files:
            f            = request.files["file"]
            content_type = f.content_type or ""
            raw          = _bytes_from_request_file(f)

            if "video" in content_type or f.filename.lower().endswith(
                (".mp4", ".mov", ".avi", ".mkv", ".webm")
            ):
                is_video         = True
                media_type       = "video"
                image_bytes_list = _extract_video_frames(raw)
                if not image_bytes_list:
                    return jsonify({"error": "Could not extract frames from video."}), 422
            else:
                media_type       = "image"
                image_bytes_list = [raw]

        # ── 2. JSON payload ──────────────────────────────────────────────────
        elif request.is_json:
            payload    = request.get_json()
            media_type = payload.get("type", "image").lower()
            is_video   = media_type == "video"

            if "base64" in payload:
                raw              = _bytes_from_base64(payload["base64"])
                image_bytes_list = _extract_video_frames(raw) if is_video else [raw]
            elif "url" in payload:
                if is_video:
                    return jsonify({"error": "URL-based video scanning not supported."}), 400
                image_bytes_list = [_bytes_from_url(payload["url"])]
            else:
                return jsonify({"error": "Provide 'file', 'base64', or 'url'."}), 400

        else:
            return jsonify({"error": "No media provided."}), 400

        # ── 3. Scan ──────────────────────────────────────────────────────────
        frame_results = []
        worst_score   = 0.0
        any_blocked   = False

        for idx, img_bytes in enumerate(image_bytes_list):
            res          = _scan_image_bytes(img_bytes)
            res["frame"] = idx
            frame_results.append(res)
            if res["score"] > worst_score:
                worst_score = res["score"]
            if res["blocked"]:
                any_blocked = True

        # ── 4. Response ──────────────────────────────────────────────────────
        body = {
            "blocked":    any_blocked,
            "mediaType":  media_type,
            "label":      "nsfw" if any_blocked else "normal",
            "score":      round(worst_score, 4),
            "confidence": f"{round(worst_score * 100, 1)}%",
            "reason":     (
                "Offensive, explicit, or sensitive content detected — upload blocked."
                if any_blocked else "Content appears safe."
            ),
        }

        if is_video:
            body["frames"] = frame_results

        return jsonify(body)

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print("Media detection error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)