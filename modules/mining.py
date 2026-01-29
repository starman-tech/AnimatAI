# File: Projet_IA/modules/mining.py
# Unchanged
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

# ==========================================
# 1. DEPENDANCES
# ==========================================
def install_deps():
    pass  # In requirements.txt

# ==========================================
# 2. CONFIGURATION OPTIMISÉE
# ==========================================
BASE_DIR = os.getcwd()
FINAL_DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_TRACKER = os.path.join(BASE_DIR, "video_manager.json")

TEMP_LOCAL_DIR = os.path.join(BASE_DIR, "temp_data_cache")
MODEL_VOSK_PATH = os.path.join(BASE_DIR, "model_fr")
MIN_DURATION_SEC = 4.0
FRAME_SKIP = 2

BS_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
]

def load_tracker():
    if not os.path.exists(JSON_TRACKER):
        with open(JSON_TRACKER, 'w') as f: json.dump([], f)
        return []
    try:
        with open(JSON_TRACKER, 'r') as f: return json.load(f)
    except: return []

def save_tracker(data):
    with open(JSON_TRACKER, 'w') as f: json.dump(data, f, indent=4)

def add_links_bulk(text_block):
    data = load_tracker()
    regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    ids = re.findall(regex, text_block)
    count = 0
    for vid_id in ids:
        clean_url = f"https://www.youtube.com/watch?v={vid_id}"
        if not any(v['url'] == clean_url for v in data):
            data.append({"url": clean_url, "id": vid_id, "processed": False, "added_at": str(datetime.now()), "segments": 0})
            count += 1
            print(f"  ✅ Ajouté : {vid_id}")
    save_tracker(data)
    return count

def setup_resources():
    if not os.path.exists(FINAL_DATA_DIR): os.makedirs(FINAL_DATA_DIR)
    if not os.path.exists(TEMP_LOCAL_DIR): os.makedirs(TEMP_LOCAL_DIR)

    if not os.path.exists(MODEL_VOSK_PATH):
        print("⬇️ DL Modèle Vocal...")
        os.system("wget -q https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip -O model.zip")
        import zipfile
        with zipfile.ZipFile("model.zip", 'r') as z: z.extractall("."); os.rename(z.namelist()[0].split('/')[0], MODEL_VOSK_PATH); os.remove("model.zip")
    if not os.path.exists('face_landmarker.task'):
        os.system('wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task')

def download_video(url):
    output = "temp_miner.mp4"
    if os.path.exists(output): os.remove(output)
    opts = {'format': 'best[ext=mp4][height<=480]/best[ext=mp4]', 'outtmpl': output, 'quiet': True, 'no_warnings': True, 'extractor_args': {'youtube': {'player_client': ['android', 'web']}}}
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([url])
        return os.path.exists(output) and os.path.getsize(output) > 1000
    except: return False

def extract_full_audio_wav():
    if os.path.exists("full_audio.wav"): os.remove("full_audio.wav")
    cmd = "ffmpeg -i temp_miner.mp4 -ac 1 -ar 16000 -vn full_audio.wav -loglevel quiet"
    os.system(cmd)
    return os.path.exists("full_audio.wav")

def get_face_features_58(landmarks, blendshapes):
    bs_dict = {b.category_name: b.score for b in blendshapes}

    cx, cy = landmarks[1].x, landmarks[1].y
    eyeL, eyeR = landmarks[33], landmarks[263]
    dx, dy = eyeR.x - eyeL.x, eyeR.y - eyeL.y

    pose = [
        1.0 - cx,
        cy,
        np.sqrt(dx*dx+dy*dy)*6.0,
        0,
        (cx-(eyeL.x+eyeR.x)/2)*8,
        -np.arctan2(dy, dx)
    ]

    bs_vector = [bs_dict.get(name, 0.0) for name in BS_NAMES]

    return pose + bs_vector

def transcribe_wav_chunk(wf, start_sec, duration_sec):
    try:
        wf.setpos(int(start_sec * 16000))
        frames_to_read = int(duration_sec * 16000)
        data = wf.readframes(frames_to_read)
        from vosk import Model, KaldiRecognizer
        rec = KaldiRecognizer(Model(MODEL_VOSK_PATH), 16000)
        rec.SetWords(False)
        rec.AcceptWaveform(data)
        res = json.loads(rec.FinalResult())
        return res.get("text", "").strip()
    except: return ""

def move_cache_to_drive():
    files = os.listdir(TEMP_LOCAL_DIR)
    if not files: return 0
    count = 0
    for f in files:
        src = os.path.join(TEMP_LOCAL_DIR, f)
        dst = os.path.join(FINAL_DATA_DIR, f)
        try:
            shutil.move(src, dst)
            count += 1
        except: pass
    return count

def run_miner():
    setup_resources()

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from tqdm import tqdm

    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    tracker = load_tracker()
    queue = [v for v in tracker if not v['processed']]
    if not queue: print("✅ Tout est à jour."); return

    print(f"🚀 TURBO MINING 58-HD : {len(queue)} vidéos")

    with tqdm(total=len(queue), desc="FILE D'ATTENTE", unit="vidéo") as global_bar:
        for vid in queue:
            url = vid['url']; vid_id = vid['id']
            gc.collect()

            if os.path.exists(TEMP_LOCAL_DIR): shutil.rmtree(TEMP_LOCAL_DIR)
            os.makedirs(TEMP_LOCAL_DIR)

            global_bar.set_postfix(status=f"DL {vid_id}...")
            if not download_video(url): global_bar.write(f"❌ Erreur DL"); global_bar.update(1); continue

            global_bar.set_postfix(status="Extraction Audio...")
            if not extract_full_audio_wav(): global_bar.write(f"❌ Erreur Audio"); continue

            try:
                cap = cv2.VideoCapture("temp_miner.mp4")
                fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                wf = wave.open("full_audio.wav", "rb")

                with tqdm(total=total_frames, desc=f"⚡ {vid_id}", leave=False, unit="fr") as video_bar:
                    curr_vecs = []; start_t = 0.0; frame_i = 0; seg_count = 0; last_ts = -1

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break

                        t_sec = frame_i / fps
                        ts_ms = int(t_sec * 1000)
                        if ts_ms <= last_ts: ts_ms = last_ts + 1
                        last_ts = ts_ms

                        if frame_i % FRAME_SKIP != 0:
                            frame_i += 1; video_bar.update(1); continue

                        h, w = frame.shape[:2]
                        if w > 480: frame = cv2.resize(frame, (480, int(h*(480/w))))

                        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        res = detector.detect_for_video(img, ts_ms)

                        valid = False
                        if len(res.face_landmarks) == 1:
                            valid = True
                            try: shapes = res.face_blendshapes[0].categories
                            except: shapes = res.face_blendshapes[0]
                            curr_vecs.append(get_face_features_58(res.face_landmarks[0], shapes))

                        if not valid:
                            if len(curr_vecs) > ((fps/FRAME_SKIP) * MIN_DURATION_SEC):
                                end_t = t_sec
                                duration = end_t - start_t
                                txt = transcribe_wav_chunk(wf, start_t, duration)

                                if len(txt) > 3:
                                    idx_smile_l = 6 + 44
                                    idx_smile_r = 6 + 45
                                    smile_avg = np.mean([(v[idx_smile_l] + v[idx_smile_r])/2 for v in curr_vecs])

                                    emo = 1 if smile_avg > 0.3 else (2 if smile_avg < 0.05 else 0)

                                    fname = f"{TEMP_LOCAL_DIR}/yt_{vid['id']}_{int(start_t)}.json"
                                    with open(fname, 'w') as f: json.dump({"text":txt, "emotion":emo, "frames":curr_vecs}, f)

                                    seg_count += 1
                                    video_bar.set_postfix(saved=seg_count, last=txt[:15])

                            curr_vecs = []
                            start_t = t_sec + (1/fps)

                        elif len(curr_vecs) == 1: start_t = t_sec

                        frame_i += 1; video_bar.update(FRAME_SKIP)

                cap.release(); wf.close()

                global_bar.set_postfix(status="Sync...")
                n_moved = move_cache_to_drive()

                for v in tracker:
                    if v['url'] == url: v['processed'] = True; v['segments'] = seg_count
                save_tracker(tracker)

                global_bar.update(1)
                global_bar.set_postfix(status="Terminé", segs=seg_count)

            except Exception as e: global_bar.write(f"❌ Crash {vid_id}: {e}"); global_bar.update(1)

            if os.path.exists("temp_miner.mp4"): os.remove("temp_miner.mp4")
            if os.path.exists("full_audio.wav"): os.remove("full_audio.wav")

def run_miner_menu():
    while True:
        print("\n=== MINER V10.5 (58 BLENDSHAPES HD) ===")
        s = load_tracker(); todo = sum(1 for x in s if not x['processed'])
        print(f"📊 En attente : {todo}")
        print("1. ➕ Ajouter Liens")
        print("2. 🚀 Lancer TURBO")
        print("3. 🚪 Quitter")
        c = input("Choix: ")
        if c == '1':
            txt = input("Liens: ")
            if txt: n = add_links_bulk(txt); print(f"✅ {n} ajoutés.")
        elif c == '2': run_miner()
        elif c == '3': break

if __name__ == "__main__":
    run_miner_menu()