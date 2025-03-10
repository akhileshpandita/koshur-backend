import os
import tempfile
import requests
import torch
import librosa  # Add this import at the top if it's missin
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Initialize Resemblyzer model
encoder = VoiceEncoder()  # ✅ Uses half-precision floats (less memory)

# DSAL audio links keyed by word
dsal_audio_links = {
    "Apple": "https://dsal.uchicago.edu/dictionaries/hassan/audio/04373.mp3",
    "Mother": "https://dsal.uchicago.edu/dictionaries/hassan/audio/02574.mp3",
    "Thief": "https://dsal.uchicago.edu/dictionaries/hassan/audio/04377.mp3",
    "Work": "https://dsal.uchicago.edu/dictionaries/hassan/audio/01758.mp3",
    "Apricot": "https://dsal.uchicago.edu/dictionaries/hassan/audio/04325.mp3",
    "Cucumber": "https://dsal.uchicago.edu/dictionaries/hassan/audio/02376.mp3",
    "Stupid": "https://dsal.uchicago.edu/dictionaries/hassan/audio/00643.mp3",
    "Darkness": "https://dsal.uchicago.edu/dictionaries/hassan/audio/00220.mp3",
    "Ladder": "https://dsal.uchicago.edu/dictionaries/hassan/audio/01521.mp3",
    "Finger": "https://dsal.uchicago.edu/dictionaries/hassan/audio/04402.mp3",
    "Blind": "https://dsal.uchicago.edu/dictionaries/hassan/audio/02989.mp3",
    "Bread": "https://dsal.uchicago.edu/dictionaries/hassan/audio/04354.mp3",
    "Mouth": "https://dsal.uchicago.edu/dictionaries/hassan/audio/00811.mp3",
    "Bridge": "https://dsal.uchicago.edu/dictionaries/hassan/audio/01880.mp3"
}

# Create a folder to cache DSAL audio files
dsal_folder = "dsal_audio"
os.makedirs(dsal_folder, exist_ok=True)

def get_dsal_audio(word):
    if word not in dsal_audio_links:
        return None
    dsal_path = os.path.join(dsal_folder, f"{word}.mp3")
    if not os.path.exists(dsal_path):
        url = dsal_audio_links[word]
        response = requests.get(url)
        if response.status_code == 200:
            with open(dsal_path, "wb") as f:
                f.write(response.content)
        else:
            return None
    return dsal_path

def load_audio_optimized(audio_path):
    """
    Load only the first 5 seconds of audio to reduce memory usage.
    """
    wav, sr = librosa.load(audio_path, sr=16000, mono=True, duration=5)
    return wav


def compute_audio_similarity(user_audio_path, dsal_audio_path):
    user_wav = load_audio_optimized(user_audio_path)
    dsal_wav = load_audio_optimized(dsal_audio_path)

    # ✅ Convert embeddings to lower precision manually
    user_embed = torch.tensor(encoder.embed_utterance(user_wav)).half().numpy()
    dsal_embed = torch.tensor(encoder.embed_utterance(dsal_wav)).half().numpy()

    similarity = np.dot(user_embed, dsal_embed) / (np.linalg.norm(user_embed) * np.linalg.norm(dsal_embed))
    return round(similarity * 100, 2)


@app.route("/compare_audio", methods=["POST"])
def compare_audio():
    word = request.form.get("word")
    if not word or word not in dsal_audio_links:
        return jsonify({"error": "Invalid or missing word parameter"}), 400
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    # Determine the file extension from the uploaded file's name
    _, ext = os.path.splitext(audio_file.filename)
    if not ext:
        ext = ".webm"  # Fallback to .webm if no extension is found

    # Save uploaded audio temporarily with the correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        audio_file.save(tmp)
        user_audio_path = tmp.name

    dsal_audio_path = get_dsal_audio(word)
    if dsal_audio_path is None:
        os.remove(user_audio_path)
        return jsonify({"error": "Failed to retrieve DSAL audio"}), 500

    try:
        score = compute_audio_similarity(user_audio_path, dsal_audio_path)
    except Exception as e:
        os.remove(user_audio_path)
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(user_audio_path)

    if score > 75:
        feedback = "Very Good"
    elif score >= 50:
        feedback = "Good but needs a little practice"
    else:
        feedback = "Try Again"

    return jsonify({"score": score, "feedback": feedback})

# Default route for testing
@app.route("/")
def index():
    return "KOSHUR Backend is running."

if __name__ == "__main__":
    app.run(debug=True)
