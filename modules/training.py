# File: Projet_IA/modules/training.py
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import time
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================
BATCH_SIZE = 8
EPOCHS = 250
LEARNING_RATE = 0.001
MOUTH_BOOST_FACTOR = 1.8  # On y va fort pour l'articulation

# Local paths (relative to this script's directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of training.py
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))  # Parent directory (e.g., AnimatAI root)
DATA_DIR = os.path.join(BASE_DIR, 'data')  # Adjust folder name if it's "data"
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'face_model.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Liste pour repérer les indices de la bouche (Même ordre que le Mining)
BS_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
]

# On calcule quels indices (parmi les 58) concernent la bouche pour les booster
# Les 6 premiers sont la POSE, donc on décale de 6
MOUTH_INDICES = [i + 6 for i, name in enumerate(BS_NAMES) if "mouth" in name or "jaw" in name]
JAW_OPEN_INDEX = 6 + BS_NAMES.index("jawOpen")  # Index précis de l'ouverture mâchoire (31)

# ==========================================
# 2. MENU SELECTION
# ==========================================
def get_ts(f):
    try: return int(f.split('_')[-1].replace('.json',''))
    except: return 0

def select_data():
    files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    if not files: print("❌ Pas de données."); return []

    print(f"\n📂 {len(files)} fichiers disponibles.")
    print("1. Tout")
    print("2. Dernières 24h")
    print("3. Seulement Manuels (Pas YouTube)")
    c = input("👉 Choix : ")

    if c == '2':
        lim = time.time() - 86400
        files = [f for f in files if get_ts(f) > lim]
    elif c == '3':
        files = [f for f in files if "manual_" in f]

    print(f"✅ {len(files)} fichiers retenus.")
    return files

# ==========================================
# 3. DATASET INTELLIGENT
# ==========================================
class RobotDataset(Dataset):
    def __init__(self, file_list):
        self.samples = []
        print("🔄 Chargement...")

        for fpath in file_list:
            try:
                with open(fpath, 'r') as f: data = json.load(f)
                frames = np.array(data['frames'], dtype=np.float32)

                # Vérification dimension (On veut du 58 !)
                if frames.shape[1] != 58:
                    continue

                # FILTRE "MOU"
                max_open = np.max(frames[:, JAW_OPEN_INDEX])
                if "yt_" in fpath and max_open < 0.05: continue

                # BOOST EXPRESSIVITÉ CIBLÉ
                for idx in MOUTH_INDICES:
                    frames[:, idx] *= MOUTH_BOOST_FACTOR

                # Clamp global
                frames[:, 6:] = np.clip(frames[:, 6:], 0.0, 1.0)

                # INPUT (Texte)
                txt = data['text'].lower()
                emo = int(data['emotion'])
                seq_len = len(frames)
                input_seq = np.zeros((seq_len, 132), dtype=np.float32)
                chars = list(txt)
                if not chars: continue

                stretch = seq_len / len(chars)
                for i in range(seq_len):
                    c = chars[min(int(i/stretch), len(chars)-1)]
                    idx = ord(c) if ord(c)<128 else 0
                    input_seq[i][idx] = 1.0
                    input_seq[i][128+emo] = 1.0

                self.samples.append((input_seq, frames))
            except: pass

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return torch.FloatTensor(self.samples[i][0]), torch.FloatTensor(self.samples[i][1])

def collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    inputs, targets = zip(*batch)
    max_len = inputs[0].shape[0]
    px = torch.zeros(len(inputs), max_len, 132)
    py = torch.zeros(len(inputs), max_len, 58)
    for i, (x, y) in enumerate(zip(inputs, targets)):
        l = x.shape[0]
        px[i, :l, :] = x
        py[i, :l, :] = y
    return px, py

# ==========================================
# 4. MODÈLE 58-OUT
# ==========================================
class FaceModel(nn.Module):
    def __init__(self, i, h, o):
        super(FaceModel, self).__init__()
        self.lstm = nn.LSTM(i, h, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(h, o)
    def forward(self, x): return self.fc(self.lstm(x)[0])

def train():
    files = select_data()
    if not files: return

    ds = RobotDataset(files)
    if len(ds) == 0: print("❌ Aucune donnée compatible 58-points trouvée."); return

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = FaceModel(132, 256, 58).to(device)

    if os.path.exists(MODEL_SAVE_PATH):
        try:
            saved_state = torch.load(MODEL_SAVE_PATH, map_location=device)
            if saved_state['fc.weight'].shape[0] == 58:
                model.load_state_dict(saved_state)
                print("🧠 Modèle 58pts existant chargé.")
            else:
                print("⚠️ Ancien modèle ignoré.")
        except: pass

    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit = nn.MSELoss()

    print("🔥 Entraînement 58-Blendshapes...")
    model.train()

    for ep in range(EPOCHS):
        loss_val = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_val += loss.item()

        if (ep+1)%10==0: print(f"Epoch {ep+1} | Loss: {loss_val/len(dl):.5f}")

    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Modèle 58-HD sauvegardé localement.")

if __name__ == "__main__":
    train()