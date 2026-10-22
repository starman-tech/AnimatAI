# File: Projet_IA/modules/website.py
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import librosa
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

app = Flask(__name__, template_folder='/home/matte/Desktop/HARDWARE/PROJECT_AnimatAI/A2B/modules/templates')
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models/animatai_audio_model.pth")
EMO_MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# --- CHARGEMENT DES IA ---
print("⏳ Chargement des IA (Émotion + Robot)...")
emo_extractor = Wav2Vec2FeatureExtractor.from_pretrained(EMO_MODEL_ID)
emo_model = AutoModelForAudioClassification.from_pretrained(EMO_MODEL_ID).to(device)
emo_model.eval()

class AnimatAIBrain(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=256, output_dim=58):
        super(AnimatAIBrain, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=3, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim), nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

active_model = None
def load_robot_model():
    global active_model
    if os.path.exists(MODEL_PATH):
        try:
            active_model = AnimatAIBrain().to(device)
            active_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            active_model.eval()
            print("✅ Modèle Robot 58-HD (Audio) chargé !")
        except Exception as e: print(f"❌ Erreur modèle : {e}")

load_robot_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_data', methods=['POST'])
def save_data():
    try:
        json_str = request.form.get('json_data')
        audio_file = request.files.get('audio_blob')
        data = json.loads(json_str)
        timestamp = int(time.time())
        base_name = f"manual_{timestamp}"
        
        # Save JSON
        with open(os.path.join(DATA_DIR, base_name + ".json"), 'w') as f:
            json.dump(data, f)
        # Save WAV
        if audio_file:
            audio_file.save(os.path.join(DATA_DIR, base_name + ".wav"))
        return jsonify({"status": "success"})
    except Exception as e: return jsonify({"error": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    if active_model is None: 
        return jsonify({"error": "Le fichier animatai_audio_model.pth est introuvable dans le dossier models/"})
    
    try:
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "Le serveur n'a reçu aucun fichier audio."})

        # Charger l'audio avec librosa
        # Utilise soundfile en backend pour éviter les erreurs de format
        y, sr = librosa.load(audio_file, sr=16000)
        
        if len(y) < 1600: # Moins de 0.1s
            return jsonify({"error": "Audio trop court."})

        # 1. IA Emotion (10 curseurs)
        inputs = emo_extractor(y, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = emo_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # S'assurer que le vecteur fait exactement 10
        emo_vector = np.zeros(10)
        size = min(len(probs), 10)
        emo_vector[:size] = probs[:size]
        
        # 2. Audio MFCC (40)
        # Calcul du nombre de frames attendu à 30 FPS
        duration = len(y) / 16000
        target_frames = max(1, int(duration * 30))
        
        # Extraction MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40, hop_length=int(16000/30)).T
        
        # On aligne le nombre de frames MFCC avec le vecteur d'émotion
        mfcc = mfcc[:target_frames]
        if len(mfcc) < target_frames:
            # Padding si trop court
            pad_width = target_frames - len(mfcc)
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='edge')
        
        # 3. Fusion (50)
        emo_seq = np.tile(emo_vector, (len(mfcc), 1))
        combined = np.concatenate((mfcc, emo_seq), axis=-1)
        
        # 4. Prédiction
        input_tensor = torch.FloatTensor([combined]).to(device)
        with torch.no_grad():
            out = active_model(input_tensor)
            # Retourner les frames en liste
            return jsonify({"frames": out.cpu().numpy()[0].tolist()})

    except Exception as e:
        import traceback
        traceback.print_exc() # Affiche l'erreur complète dans ton terminal noir
        return jsonify({"error": str(e)})
if __name__ == "__main__":
    app.run(port=5000)