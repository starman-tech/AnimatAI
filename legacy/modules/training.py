import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import librosa
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

# ==========================================
# 1. CONFIGURATION & IA EMOTION
# ==========================================
BATCH_SIZE = 4
EPOCHS = 150
LEARNING_RATE = 0.0005
N_MFCC = 40
N_EMOTIONS = 10 # Les 10 curseurs (probabilités)
INPUT_DIM = N_MFCC + N_EMOTIONS 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle d'émotion (Prédit les intensités)
# Ce modèle de base gère 8 émotions, nous l'adapterons à 10 ou utiliserons ses sorties probit
EMO_MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
emo_extractor = Wav2Vec2FeatureExtractor.from_pretrained(EMO_MODEL_ID)
emo_model = AutoModelForAudioClassification.from_pretrained(EMO_MODEL_ID).to(device)
emo_model.eval()

# Chemins
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data') 
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'animatai_audio_model.pth')

# ==========================================
# 2. UTILITAIRES AUDIO
# ==========================================

def get_audio_features(audio_path, target_frames):
    """ Extrait les MFCC et les synchronise avec le nombre de frames JSON """
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Extraction MFCC
    # On ajuste le hop_length pour que le nombre de MFCC corresponde aux frames du robot
    duration = librosa.get_duration(y=y, sr=sr)
    hop_length = int((sr * duration) / target_frames)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=hop_length)
    mfcc = mfcc[:, :target_frames].T # Shape: (target_frames, 40)
    
    # Extraction des 10 Curseurs d'Émotions (via Wav2Vec2)
    inputs = emo_extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = emo_model(**inputs).logits
        # On utilise Softmax pour transformer en "intensités" (0 à 1)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
    # Si le modèle a moins de 10 classes, on complète avec du neutre/vide
    emo_vector = np.zeros(N_EMOTIONS)
    emo_vector[:len(probs)] = probs
    
    # On duplique le vecteur d'émotion pour chaque frame (émotion constante sur l'audio)
    # Dans une version avancée, on pourrait le faire par chunk de 100ms
    emo_sequence = np.tile(emo_vector, (target_frames, 1))
    
    return mfcc, emo_sequence

# ==========================================
# 3. DATASET AUDIO + JSON
# ==========================================

class AudioRobotDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        
        print(f"🔄 Préparation de l'audio et des émotions pour {len(json_files)} fichiers...")

        for j_path in json_files:
            audio_path = j_path.replace(".json", ".wav") # Cherche le .wav associé
            if not os.path.exists(audio_path): continue
            
            try:
                with open(j_path, 'r') as f: data = json.load(f)
                frames = np.array(data['frames'], dtype=np.float32)
                target_frames = len(frames)
                
                # Récupère l'audio MFCC + Les curseurs d'émotions
                mfcc, emotions = get_audio_features(audio_path, target_frames)
                
                # Fusion (40 MFCC + 10 Emo = 50 dimensions)
                combined_input = np.concatenate((mfcc, emotions), axis=-1)
                
                self.samples.append((combined_input, frames))
            except Exception as e:
                print(f"⚠️ Erreur sur {j_path}: {e}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): 
        return torch.FloatTensor(self.samples[i][0]), torch.FloatTensor(self.samples[i][1])

def collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    inputs, targets = zip(*batch)
    max_len = inputs[0].shape[0]
    px = torch.zeros(len(inputs), max_len, INPUT_DIM)
    py = torch.zeros(len(inputs), max_len, 58)
    for i, (x, y) in enumerate(zip(inputs, targets)):
        l = x.shape[0]
        px[i, :l, :] = x
        py[i, :l, :] = y
    return px, py

# ==========================================
# 4. MODÈLE LSTM MULTIMODAL
# ==========================================

class AnimatAIBrain(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AnimatAIBrain, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=3, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid() # Mouvements entre 0 et 1
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

# ==========================================
# 5. BOUCLE D'ENTRAÎNEMENT
# ==========================================

def train():
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    
    ds = AudioRobotDataset(DATA_DIR)
    if len(ds) == 0: 
        print("❌ Aucune paire (JSON + WAV) trouvée.")
        return

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model = AnimatAIBrain(INPUT_DIM, 256, 58).to(device)
    
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit = nn.MSELoss()

    print(f"🔥 Entraînement sur {INPUT_DIM} entrées (Audio + 10 Curseurs)...")
    
    for ep in range(EPOCHS):
        total_loss = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        if (ep+1) % 5 == 0:
            print(f"Epoch {ep+1}/{EPOCHS} | Loss: {total_loss/len(dl):.6f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Modèle sauvegardé sous : {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()