# File: Projet_IA/modules/mining.py
import os
import sys
import subprocess
import cv2
import json
import numpy as np
import shutil
import re
import time
import gc
from datetime import datetime
import wave
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = os.getcwd()
FINAL_DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_TRACKER = os.path.join(BASE_DIR, "video_manager.json")
TEMP_LOCAL_DIR = os.path.join(BASE_DIR, "temp_data_cache")

MIN_DURATION_SEC = 2.5 # Un segment doit faire au moins 2.5s
FRAME_SKIP = 2 # Traiter 1 image sur 2 pour la vitesse

BS_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
]

# ==========================================
# 2. UTILITAIRES
# ==========================================

def load_tracker():
    if not os.path.exists(JSON_TRACKER):
        with open(JSON_TRACKER, 'w') as f: json.dump([], f)
        return []
    try:
        with open(JSON_TRACKER, 'r') as f: return json.load(f)
    except: return []

def save_tracker(data):
    with open(JSON_TRACKER, 'w') as f: json.dump(data, f, indent=4)

def setup_resources():
    if not os.path.exists(FINAL_DATA_DIR): os.makedirs(FINAL_DATA_DIR)
    if not os.path.exists(TEMP_LOCAL_DIR): os.makedirs(TEMP_LOCAL_DIR)

    # Vérification FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("❌ ERREUR : FFmpeg est requis !")
        sys.exit(1)
    
    if not os.path.exists('face_landmarker.task'):
        print("⬇️ Téléchargement du modèle de tracking facial...")
        os.system('wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task')

# ==========================================
# 3. MOTEUR DE TÉLÉCHARGEMENT & AUDIO
# ==========================================

def download_video(url):
    output = "temp_miner.mp4"
    if os.path.exists(output): os.remove(output)
    print(f"  📥 Téléchargement de la vidéo...")
    opts = {'format': 'best[ext=mp4][height<=480]', 'outtmpl': output, 'quiet': True, 'no_warnings': True}
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([url])
        return os.path.exists(output)
    except Exception as e:
        print(f"  ❌ Erreur yt-dlp : {e}")
        return False

def extract_full_audio_wav():
    if os.path.exists("full_audio.wav"): os.remove("full_audio.wav")
    cmd = ["ffmpeg", "-i", "temp_miner.mp4", "-ac", "1", "-ar", "16000", "-vn", "full_audio.wav", "-loglevel", "quiet"]
    subprocess.run(cmd)
    return os.path.exists("full_audio.wav")

def save_audio_segment(start_sec, duration_sec, output_path):
    cmd = f"ffmpeg -y -ss {start_sec} -t {duration_sec} -i full_audio.wav -ac 1 -ar 16000 {output_path} -loglevel quiet"
    os.system(cmd)

# ==========================================
# 4. ANALYSE FACIALE
# ==========================================

def get_face_features_58(landmarks, blendshapes):
    bs_dict = {b.category_name: b.score for b in blendshapes}
    cx, cy = landmarks[1].x, landmarks[1].y
    eyeL, eyeR = landmarks[33], landmarks[263]
    dx, dy = eyeR.x - eyeL.x, eyeR.y - eyeL.y
    pose = [1.0 - cx, cy, np.sqrt(dx*dx+dy*dy)*6.0, 0, (cx-(eyeL.x+eyeR.x)/2)*8, -np.arctan2(dy, dx)]
    bs_vector = [bs_dict.get(name, 0.0) for name in BS_NAMES]
    return pose + bs_vector

# ==========================================
# 5. CORE MINER V12
# ==========================================

def run_miner():
    setup_resources()
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
            output_face_blendshapes=True,
            running_mode=vision.RunningMode.VIDEO)
    )

    tracker = load_tracker()
    queue = [v for v in tracker if not v['processed']]
    if not queue: print("✅ Tout est déjà traité."); return

    print(f"🚀 Lancement du Mining sur {len(queue)} vidéos...")

    for vid in queue:
        url = vid['url']; vid_id = vid['id']
        print(f"\n🎬 Traitement de : {vid_id}")
        
        if not download_video(url): continue
        if not extract_full_audio_wav(): continue

        cap = cv2.VideoCapture("temp_miner.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        curr_vecs = []; start_t = 0.0; frame_i = 0; seg_count = 0; last_ts = -1
        face_detected_at_least_once = False

        # BARRE DE PROGRESSION LOCALE
        pbar = tqdm(total=total_frames, desc="  ⚡ Analyse Vidéo", unit="fr")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            t_sec = frame_i / fps
            ts_ms = int(t_sec * 1000)
            if ts_ms <= last_ts: ts_ms = last_ts + 1
            last_ts = ts_ms

            if frame_i % FRAME_SKIP == 0:
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = detector.detect_for_video(img, ts_ms)

                if len(res.face_landmarks) == 1:
                    face_detected_at_least_once = True
                    if not curr_vecs: start_t = t_sec
                    
                    shapes = res.face_blendshapes[0].categories if hasattr(res.face_blendshapes[0], 'categories') else res.face_blendshapes[0]
                    curr_vecs.append(get_face_features_58(res.face_landmarks[0], shapes))
                else:
                    # Visage perdu ou absent : vérifier si on doit sauvegarder le segment accumulé
                    if len(curr_vecs) > ((fps/FRAME_SKIP) * MIN_DURATION_SEC):
                        duration = t_sec - start_t
                        base_name = f"yt_{vid_id}_{int(start_t)}"
                        
                        # Chemins
                        json_path = os.path.join(TEMP_LOCAL_DIR, base_name + ".json")
                        wav_path = os.path.join(TEMP_LOCAL_DIR, base_name + ".wav")

                        # Émotion simple pour le JSON (Joie si sourire > 30%)
                        idx_smile = 6 + 44 
                        smile_avg = np.mean([v[idx_smile] for v in curr_vecs])
                        emo = 1 if smile_avg > 0.3 else 0

                        # Sauvegarde
                        with open(json_path, 'w') as f:
                            json.dump({"emotion":emo, "frames":curr_vecs, "src":vid_id}, f)
                        save_audio_segment(start_t, duration, wav_path)
                        
                        seg_count += 1
                        pbar.set_postfix(segments=seg_count)
                    
                    curr_vecs = []

            frame_i += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        if not face_detected_at_least_once:
            print("  ⚠️ Attention : Aucun visage n'a été détecté sur cette vidéo !")
        
        # Déplacement des fichiers temporaires vers data/
        files = os.listdir(TEMP_LOCAL_DIR)
        for f in files:
            shutil.move(os.path.join(TEMP_LOCAL_DIR, f), os.path.join(FINAL_DATA_DIR, f))
        
        # Mise à jour du tracker
        vid['processed'] = True; vid['segments'] = seg_count
        save_tracker(tracker)
        print(f"  ✅ Terminé : {seg_count} segments extraits.")

        # Nettoyage
        if os.path.exists("temp_miner.mp4"): os.remove("temp_miner.mp4")
        if os.path.exists("full_audio.wav"): os.remove("full_audio.wav")
        gc.collect()

def add_links_bulk(text_block):
    data = load_tracker()
    regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    ids = re.findall(regex, text_block)
    count = 0
    for vid_id in ids:
        clean_url = f"https://www.youtube.com/watch?v={vid_id}"
        if not any(v['id'] == vid_id for v in data):
            data.append({"url": clean_url, "id": vid_id, "processed": False, "added_at": str(datetime.now()), "segments": 0})
            count += 1
    save_tracker(data)
    return count

def run_miner_menu():
    while True:
        print("\n=== MINER V12 (HD AUDIO + NO VOSK) ===")
        s = load_tracker(); todo = sum(1 for x in s if not x['processed'])
        print(f"📊 Vidéos en attente : {todo}")
        print("1. ➕ Ajouter Liens")
        print("2. 🚀 Lancer MINING")
        print("3. 🚪 Quitter")
        c = input("Choix: ")
        if c == '1':
            txt = input("Collez les liens : ")
            if txt: n = add_links_bulk(txt); print(f"✅ {n} ajoutés.")
        elif c == '2': run_miner()
        elif c == '3': break

if __name__ == "__main__":
    run_miner_menu()