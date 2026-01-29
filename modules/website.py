# File: Projet_IA/modules/website.py
# Unchanged
import os
import sys
import subprocess
import threading
import time
import random
import json
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import torch
import torch.nn as nn
from vosk import Model, KaldiRecognizer

app = Flask(__name__, template_folder='templates')
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models/face_model.pth")

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

print(f"📂 Dossier données : {DATA_DIR}")
print(f"🧠 Chemin modèle cherché : {MODEL_PATH}")

class FaceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FaceModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

active_model = None

def load_model_logic():
    global active_model
    if os.path.exists(MODEL_PATH):
        try:
            temp_model = FaceModel(132, 256, 58).to(device)
            state = torch.load(MODEL_PATH, map_location=device)

            if state['fc.weight'].shape[0] == 58:
                temp_model.load_state_dict(state)
                temp_model.eval()
                active_model = temp_model
                print(f"✅ SUCCÈS : Modèle 58 chargé !")
                return True
            else:
                print(f"❌ ERREUR : Le modèle trouvé n'est pas un modèle 58 points.")
        except Exception as e:
            print(f"⚠️ Erreur fichier: {e}")
    else:
        print(f"❌ FICHIER MANQUANT : {MODEL_PATH}")
    return False

load_model_logic()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_data', methods=['POST'])
def save_data():
    data = request.json
    emo_idx = int(data.get('emotion', 0))
    emo_label = ["neutre", "joie", "triste", "colere"][emo_idx]

    timestamp = int(time.time())
    filename = f"manual_{emo_label}_{timestamp}.json"
    filepath = os.path.join(DATA_DIR, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        return jsonify({"status": "saved", "path": filepath})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    global active_model

    if active_model is None:
        load_model_logic()

    if active_model is None:
        return jsonify({"error": f"Modèle introuvable dans {MODEL_PATH}"})

    d = request.json
    txt = d.get('text', '').lower()
    emo = int(d.get('emotion', 0))

    seq = []
    frames_per_char = 4
    for c in txt:
        v = np.zeros(132)
        idx = ord(c) if ord(c) < 128 else 0
        v[idx] = 1.0
        v[128 + emo] = 1.0
        for _ in range(frames_per_char): seq.append(v)

    for _ in range(15):
        v = np.zeros(132); v[128+emo] = 1.0; seq.append(v)

    if len(seq) == 0: return jsonify({"frames": []})

    try:
        with torch.no_grad():
            out = active_model(torch.FloatTensor([seq]).to(device))
            return jsonify({"frames": out.cpu().numpy()[0].tolist()})
    except Exception as e: return jsonify({"error": str(e)})

html_code = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>K-VRC Robot V3 (HD Glow)</title>
    <style>
        body { margin: 0; background-color: #050505; color: #ff8800; font-family: 'Courier New', monospace; overflow: hidden; display: flex; justify-content: center; align-items: center; height: 100vh; }
        canvas { width: 100%; height: 100%; display: block; background: radial-gradient(circle, #1a1a1a 0%, #000000 90%); }
        video { position: absolute; top: 20px; right: 20px; width: 150px; opacity: 0; pointer-events: none; }
        #hud { position: absolute; bottom: 30px; left: 30px; font-size: 14px; color: #ffaa00; text-shadow: 0 0 10px #ff5500; pointer-events: none; line-height: 1.5; }
        .controls-panel { position: absolute; top: 20px; left: 20px; width: 300px; background: rgba(0, 0, 0, 0.9); padding: 15px; border: 1px solid #333; border-radius: 5px; z-index: 50; }
        button { width: 100%; background: #222; color: #ff8800; border: 1px solid #ff5500; padding: 10px; cursor: pointer; font-weight: bold; margin-bottom: 5px; }
        button:hover { background: #ff5500; color: #000; }
        textarea, select { width: 100%; background: #111; color: #fff; border: 1px solid #333; margin-bottom: 5px; box-sizing: border-box; }
        #guided-overlay { display: none; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.95); z-index: 100; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
        .hidden { display: none !important; }
        #guided-text { font-size: 32px; color: #ffaa00; margin: 30px; max-width: 80%; }
        #guided-emo { font-size: 20px; color: #fff; text-transform: uppercase; letter-spacing: 2px; }
    </style>
</head>
<body>
    <video id="webcam" autoplay playsinline></video>
    <canvas id="canvas"></canvas>

    <div class="controls-panel" id="main-panel">
        <button onclick="startGuidedSession()">▶ SESSION GUIDÉE (DATA V3)</button>
        <hr style="border-top:1px solid #333;">
        <select id="emotionSelect"><option value="0">Neutre</option><option value="1">Joyeux</option><option value="2">Triste</option><option value="3">Colère</option></select>
        <textarea id="scriptText" placeholder="Texte à tester..."></textarea>
        <button onclick="generateAnimation()">[ IA ] ANIMER ROBOT</button>
    </div>

    <div id="guided-overlay">
        <div id="guided-counter">PHRASE 1</div>
        <div id="guided-emo">NEUTRE</div>
        <div id="guided-text">Chargement...</div>
        <button id="guided-btn" onclick="toggleGuidedRecord()" style="width: 200px; height: 60px; font-size: 20px;">PARLER</button>
        <button onclick="exitGuidedMode()" style="background:none; border:none; color:#555; margin-top:20px;">(Quitter)</button>
    </div>
    <div id="hud">SYSTEM: ONLINE (HD 58)</div>

    <script type="module">
        import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

        const BS_NAMES = [
            "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
        ];

        const video = document.getElementById("webcam"), canvas = document.getElementById("canvas"), ctx = canvas.getContext("2d");
        const mainPanel = document.getElementById('main-panel'), guidedOverlay = document.getElementById('guided-overlay');
        let faceLandmarker, lastTime = -1, isRecording = false, recordedFrames = [], isPlayingAI = false, aiFrames = [], aiFrameIndex = 0;
        let recognition;

        const TRAINING_SET = [
            { t: "Système prêt pour l'analyse.", e: 0 },
            { t: "Je vois absolument tout.", e: 0 },
            { t: "Super, ça marche !", e: 1 },
            { t: "Oh non, c'est dommage.", e: 2 },
            { t: "Arrête ça !", e: 3 },
            { t: "A E I O U. Ba Be Bi Bo Bu.", e: 0 }
        ];
        let currentStep = 0;

        if ('webkitSpeechRecognition' in window) { recognition = new webkitSpeechRecognition(); recognition.continuous=true; recognition.lang='fr-FR'; }

        function mapBlendshapesToRobot(bsMap) {
            const v = (name) => bsMap[name] || 0.0;
            const s = {
                pose: { x: 0.5, y: 0.5, z: 1.0, roll: 0, yaw: 0 },
                eyeL: { h: 1, w: 1, squeeze: 0, y_offset: 0 },
                eyeR: { h: 1, w: 1, squeeze: 0, y_offset: 0 },
                browL: { y: 0, rot: 0 }, browR: { y: 0, rot: 0 },
                mouth: { width: 0, openY: 0, smile: 0, pucker: 0, funnel: 0 }
            };

            const calcEye = (side) => {
                let blink = v(`eyeBlink${side}`);
                let wide = v(`eyeWide${side}`);
                let squint = v(`eyeSquint${side}`);
                let h = Math.max(0.05, 1.0 - blink + (wide * 0.5) - (squint * 0.6));
                return { h: h, w: 1.0 - (squint * 0.2), squeeze: squint, y_offset: (v(`eyeLookDown${side}`) - v(`eyeLookUp${side}`)) * 0.8 };
            };
            s.eyeL = calcEye("Left"); s.eyeR = calcEye("Right");

            s.browL = { y: v("browInnerUp") - v("browDownLeft"), rot: (v("browInnerUp") * 0.5) - (v("browOuterUpLeft") * 0.3) - v("browDownLeft") };
            s.browR = { y: v("browInnerUp") - v("browDownRight"), rot: -(v("browInnerUp") * 0.5) + (v("browOuterUpRight") * 0.3) + v("browDownRight") };

            let open = Math.max(0, v("jawOpen") - v("mouthClose"));
            let wide = (v("mouthSmileLeft") + v("mouthSmileRight")) * 0.5;
            let narrow = v("mouthPucker") + v("mouthFunnel");
            let frown = (v("mouthFrownLeft") + v("mouthFrownRight")) * 0.5;

            s.mouth.openY = open;
            s.mouth.width = 1.0 + (wide * 0.4) - (narrow * 0.5) + (v("mouthStretchLeft") + v("mouthStretchRight"))*0.2;
            s.mouth.smile = wide - frown;
            s.mouth.pucker = v("mouthPucker"); s.mouth.funnel = v("mouthFunnel");
            return s;
        }

        window.startGuidedSession = function() { mainPanel.classList.add('hidden'); guidedOverlay.style.display = 'flex'; currentStep = 0; updateGuidedUI(); }
        window.exitGuidedMode = function() { guidedOverlay.style.display = 'none'; mainPanel.classList.remove('hidden'); }
        function updateGuidedUI() {
            if(currentStep >= TRAINING_SET.length) { exitGuidedMode(); alert("Merci !"); return; }
            const i = TRAINING_SET[currentStep];
            document.getElementById('guided-text').innerText = i.t;
            document.getElementById('guided-emo').innerText = ["NEUTRE","JOIE","TRISTE","COLERE"][i.e];
            document.getElementById('guided-emo').style.color = ["#aaa","#0f0","#00f","#f00"][i.e];
        }

        window.toggleGuidedRecord = async function() {
            const btn = document.getElementById('guided-btn');
            if(!isRecording) {
                isRecording = true; recordedFrames=[]; if(recognition) recognition.start(); btn.innerText="STOP";
            } else {
                isRecording = false; if(recognition) recognition.stop(); btn.innerText="ENVOI...";
                await fetch('/save_data', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ text: TRAINING_SET[currentStep].t, emotion: TRAINING_SET[currentStep].e, frames:recordedFrames }) });
                currentStep++; btn.innerText="PARLER"; updateGuidedUI();
            }
        }

        window.generateAnimation = async function() {
            const txt = document.getElementById('scriptText').value;
            const emo = parseInt(document.getElementById('emotionSelect').value);
            if(!txt) return alert("Entrez du texte !");

            const res = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text:txt, emotion:emo}) });
            const data = await res.json();

            if(data.error) { alert("ERREUR IA : " + data.error); return; }

            aiFrames = data.frames; aiFrameIndex = 0; isPlayingAI = true;
        }

        let robotState = mapBlendshapesToRobot({});
        let smoothState = JSON.parse(JSON.stringify(robotState));

        function applyAiVector(vec) {
            const pose = { x: vec[0], y: vec[1], z: vec[2], roll: vec[5], yaw: vec[4] };
            const bsMap = {};
            for(let i=0; i<BS_NAMES.length; i++) { if(i + 6 < vec.length) bsMap[BS_NAMES[i]] = vec[i+6]; }
            const mapped = mapBlendshapesToRobot(bsMap);
            mapped.pose = pose; robotState = mapped;
        }

        async function loop() {
            if (isPlayingAI) {
                if(aiFrameIndex < aiFrames.length) { applyAiVector(aiFrames[aiFrameIndex]); aiFrameIndex++; }
                else isPlayingAI = false;
            } else if (video.currentTime !== lastTime && faceLandmarker) {
                lastTime = video.currentTime;
                const res = faceLandmarker.detectForVideo(video, performance.now());
                if (res.faceLandmarks.length > 0) { processInput(res.faceLandmarks[0], res.faceBlendshapes[0].categories); }
            }
            smooth(); render(); requestAnimationFrame(loop);
        }

        function processInput(landmarks, blendshapesCategories) {
            const bsMap = {};
            blendshapesCategories.forEach(b => bsMap[b.categoryName] = b.score);
            let cx=landmarks[1].x, cy=landmarks[1].y;
            let eyeL=landmarks[33], eyeR=landmarks[263];
            let dx=eyeR.x-eyeL.x, dy=eyeR.y-eyeL.y;
            const pose = { x: 1.0 - cx, y: cy, z: Math.sqrt(dx*dx+dy*dy) * 6.0, roll: -Math.atan2(dy,dx), yaw: (cx-(eyeL.x+eyeR.x)/2)*8 };
            const mapped = mapBlendshapesToRobot(bsMap);
            mapped.pose = pose; robotState = mapped;
            if(isRecording) {
                const vec = [ pose.x, pose.y, pose.z, 0, pose.yaw, pose.roll ];
                BS_NAMES.forEach(name => vec.push(bsMap[name] || 0));
                recordedFrames.push(vec);
            }
        }

        function smooth() {
            const f = isPlayingAI ? 0.3 : 0.2;
            const lerp = (a,b) => a*(1-f) + b*f;
            const r = (curr, target) => { for(let k in target) { if(typeof target[k] === 'object') r(curr[k], target[k]); else curr[k] = lerp(curr[k], target[k]); } };
            r(smoothState, robotState);
        }

        function resize() {
            const dpr = window.devicePixelRatio || 1;
            canvas.width = window.innerWidth * dpr; canvas.height = window.innerHeight * dpr;
            canvas.style.width = window.innerWidth + 'px'; canvas.style.height = window.innerHeight + 'px';
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        window.addEventListener('resize', resize); resize();

        function render() {
            ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.restore();
            const s=smoothState;
            const w = canvas.width / (window.devicePixelRatio || 1);
            const h = canvas.height / (window.devicePixelRatio || 1);

            ctx.save();
            ctx.translate(w * s.pose.x, h * s.pose.y);
            ctx.rotate(s.pose.roll);
            ctx.scale(h * s.pose.z, h * s.pose.z);
            ctx.shadowBlur = 30; ctx.shadowColor = "#FF5500"; ctx.fillStyle = "#FFFFFF"; ctx.imageSmoothingEnabled = true;

            const rr=(c,x,y,w,h,r)=>{if(w<2*r)r=w/2;if(h<2*r)r=h/2;c.beginPath();c.moveTo(x+r,y-h/2);c.arcTo(x+w/2,y-h/2,x+w/2,y+h/2,r);c.arcTo(x+w/2,y+h/2,x-w/2,y+h/2,r);c.arcTo(x-w/2,y+h/2,x-w/2,y-h/2,r);c.arcTo(x-w/2,y-h/2,x+w/2,y-h/2,r);c.closePath()};

            [s.eyeL,s.eyeR].forEach((e,i)=>{
                ctx.save(); ctx.translate(i?0.22:-0.22,-0.15+e.y_offset*0.1);
                ctx.rotate((i?-1:1)*e.squeeze*0.2+(i?s.browR.rot:s.browL.rot)*0.5);
                let w=0.16*e.w,h=0.14*e.h;
                rr(ctx,0,0,w*2,h*2,0.04); ctx.fill();
                if(Math.abs(i?s.browR.y:s.browL.y)>0.1||e.squeeze>0.2){
                    ctx.beginPath(); rr(ctx,0,-h-0.05+(i?s.browR.y:s.browL.y)*0.05,w*2.2,0.03,0.01); ctx.fill();
                }
                ctx.restore();
            });

            ctx.save(); ctx.translate(0,0.2);
            let m=s.mouth,cw=0.35*m.width*(1-m.pucker*0.6),ch=Math.max(0.02,0.25*m.openY+m.pucker*0.1),sc=-m.smile*0.1;
            if(ch<0.04){
                ctx.lineWidth=0.03; ctx.lineCap="round"; ctx.strokeStyle="#fff";
                ctx.beginPath(); ctx.moveTo(-cw,sc); ctx.quadraticCurveTo(0,0,cw,sc); ctx.stroke();
            }else{
                ctx.translate(0,sc*0.5); rr(ctx,0,0,cw*2,ch*2,m.pucker>0.5?cw:0.05); ctx.fill();
            }
            ctx.restore(); ctx.restore();
        }

        async function setup() {
            const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
            faceLandmarker = await FaceLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`, delegate: "GPU" }, outputFaceBlendshapes: true, outputFaceLandmarks: true, runningMode: "VIDEO", numFaces: 1 });
            navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } }).then(stream => { video.srcObject = stream; video.addEventListener("loadeddata", loop); });
        }
        setup();
    </script>
</body>
</html>
"""

def run_website():
    app.run(port=5000)

if __name__ == "__main__":
    run_website()