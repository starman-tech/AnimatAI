"""Self-contained HTML demo: a 2D face driven by the model, next to the tracked ground truth.

    animatai demo --model runs/base/best.pt --split test --n 4 --out demo.html
    animatai demo --model runs/base/best.pt --audio my_voice.wav --out demo.html

Everything (audio, frames, renderer) is embedded: open the file in any browser.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np

from . import schema
from .audio import SAMPLE_RATE, load_audio
from .runtime.engine import Animator


def _wav_data_uri(audio: np.ndarray) -> str:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def _pack(frames: np.ndarray) -> list:
    return np.round(frames, 3).tolist()


def build_demo(model_path: str, out_path: str, audio_files: list[str] | None = None, processed_dir: str | None = None,
               split: str = "test", n: int = 4, style: str = "generic", emotion: str | None = None,
               max_sec: float = 20.0) -> Path:
    raw_animator = Animator(model_path, style=style, postprocess=False)
    full_animator = Animator(model_path, style=style, emotion=emotion)
    clips = []

    def add(title: str, audio: np.ndarray, target: np.ndarray | None):
        audio = audio[: int(max_sec * SAMPLE_RATE)]
        raw = raw_animator.predict_raw(audio)
        clip = {"title": title, "audio": _wav_data_uri(audio), "tracks": {}}
        if target is not None:
            clip["tracks"]["Tracker (vérité terrain)"] = _pack(target[: len(raw)])
        clip["tracks"]["Modèle brut"] = _pack(raw)
        clip["tracks"]["Modèle + vie (IA + animation)"] = _pack(full_animator.animate(audio))
        clips.append(clip)

    for path in audio_files or []:
        add(Path(path).name, load_audio(path), None)
    if processed_dir and not audio_files:
        from .data.prepare import load_manifest

        items = [i for i in load_manifest(processed_dir) if i["split"] == split and i["n_frames"] >= 240]
        by_source: dict[str, list[dict]] = {}
        for item in items:
            by_source.setdefault(item["source"], []).append(item)
        picked = []
        while len(picked) < n and any(by_source.values()):  # round-robin over speakers
            for source in sorted(by_source):
                if by_source[source] and len(picked) < n:
                    group = by_source[source]
                    picked.append(group.pop(len(group) // 2))
        for item in picked:
            add(f"{item['source']} · {split} · t={item['start_sec']:.0f}s",
                load_audio(Path(processed_dir) / "audio" / f"{item['id']}.wav"),
                np.load(Path(processed_dir) / "targets" / f"{item['id']}.npy"))
    if not clips:
        raise SystemExit("nothing to show: pass --audio files or a processed dir with clips in the chosen split")

    payload = {"fps": raw_animator.fps, "channels": list(schema.CHANNELS), "clips": clips,
               "model": str(model_path), "style": style}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_TEMPLATE.replace("/*DATA*/null", json.dumps(payload)))
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AnimatAI · démo</title>
<style>
:root{--bg:#f6f4ef;--panel:#fffdf8;--ink:#1d1b18;--mute:#6f6a60;--line:#e2ddd0;--accent:#c2410c;--skin:#f0c8a0;--skin2:#d9a877;--lip:#b4463c;--dark:#2a2320}
@media (prefers-color-scheme:dark){:root{--bg:#15130f;--panel:#1e1b16;--ink:#efe9dc;--mute:#9a9385;--line:#322d24;--accent:#fb923c;--skin:#e2b48a;--skin2:#b98a5c;--lip:#c0564b;--dark:#120e0c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 60px}
h1{font:600 26px/1.2 ui-serif,Georgia,serif;margin:0 0 4px}.sub{color:var(--mute);margin:0 0 22px;font-size:14px}
.clips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.clips button{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:7px 12px;border-radius:999px;cursor:pointer;font:inherit;font-size:13px}
.clips button[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:#fff}
.faces{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px}
.card h2{font-size:13px;font-weight:600;margin:0 0 6px;color:var(--mute);text-transform:uppercase;letter-spacing:.06em}
canvas{width:100%;height:auto;display:block;border-radius:10px}
canvas.robot{background:radial-gradient(circle,#1a1a1a 0%,#000 90%)}
.mode{margin-left:auto;display:flex;gap:6px}.mode button{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:6px 12px;border-radius:8px;cursor:pointer;font:inherit;font-size:13px}
.mode button[aria-pressed=true]{background:var(--ink);color:var(--bg)}
.bars{margin-top:8px;display:grid;grid-template-columns:86px 1fr;gap:3px 8px;font-size:11px;color:var(--mute);align-items:center}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent);width:0}
.transport{margin:16px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
audio{flex:1;min-width:240px}
.note{color:var(--mute);font-size:13px;margin-top:18px;border-top:1px solid var(--line);padding-top:12px}
</style></head><body><main>
<h1>AnimatAI — voix → visage</h1>
<p class="sub" id="sub"></p>
<div class="clips" id="clips"></div>
<div class="transport"><audio id="audio" controls></audio>
<div class="mode"><button id="m-robot" aria-pressed="true">Visage robot</button><button id="m-human" aria-pressed="false">Visage schématique</button><button id="m-life" aria-pressed="true">Couche « vivant »</button></div></div>
<div class="faces" id="faces"></div>
<p class="note">Le visage robot est le rendu d'origine du projet (mêmes formes, proportions, halo, vue miroir). L'IA pilote bouche, sourcils et tête ; la couche « vivant » ajoute ce que l'audio ne peut pas savoir : respiration, regard, clignements, accents sur les syllabes fortes, mouvements amortis par ressorts. Le tracker reste brut pour servir de référence. Le visage continue de vivre quand l'audio est en pause.</p>
</main>
<script>
const DATA = /*DATA*/null;
const CH = Object.fromEntries(DATA.channels.map((n,i)=>[n,i]));
const BARS = ["jawOpen","mouthClose","mouthFunnel","mouthPucker","mouthSmileLeft","browInnerUp","headYaw"];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const PREVIEW_T = parseFloat((location.hash.match(/t=([\d.]+)/)||[])[1]||0);
const TRACKER="Tracker (vérité terrain)", ALIVE="Modèle + vie (IA + animation)";
let clip = null, views = [], MODE = "robot", LIFE = true;

function drawFace(ctx, f, W, H){
  const g = n => f ? f[CH[n]] : 0, avg=(a,b)=>(g(a)+g(b))/2;
  ctx.clearRect(0,0,W,H); ctx.save();
  const yaw=g("headYaw"), pitch=g("headPitch"), roll=g("headRoll");
  ctx.translate(W/2 + yaw*W*0.10, H/2 - pitch*H*0.07); ctx.rotate(roll*Math.PI/4); ctx.scale(W/300, W/300);
  const skin=css("--skin"), skin2=css("--skin2"), dark=css("--dark"), lip=css("--lip");
  const jaw=g("jawOpen");
  // head
  ctx.fillStyle=skin; ctx.beginPath(); ctx.ellipse(0,-5+jaw*9,92,118+jaw*10,0,0,Math.PI*2); ctx.fill();
  // cheeks (puff / squint)
  ctx.fillStyle=skin2; ctx.globalAlpha=.25+.5*avg("cheekSquintLeft","cheekSquintRight");
  for(const s of [-1,1]){ctx.beginPath();ctx.ellipse(s*52+yaw*8,30,18+g("cheekPuff")*10,12,0,0,Math.PI*2);ctx.fill();}
  ctx.globalAlpha=1;
  // eyes
  for(const [s,side] of [[-1,"Right"],[1,"Left"]]){ // image-left eye is the subject's right
    const ex=s*38+yaw*10, ey=-32;
    const blink=g("eyeBlink"+side), wide=g("eyeWide"+side), squint=g("eyeSquint"+side);
    const open=Math.max(0.04,(1-blink)*(1+0.5*wide-0.45*squint));
    ctx.fillStyle="#fff"; ctx.beginPath(); ctx.ellipse(ex,ey,17,11*open,0,0,Math.PI*2); ctx.fill();
    const lx=(g("eyeLookOut"+side)-g("eyeLookIn"+side))*s*6, ly=(g("eyeLookDown"+side)-g("eyeLookUp"+side))*5;
    ctx.save(); ctx.beginPath(); ctx.ellipse(ex,ey,17,11*open,0,0,Math.PI*2); ctx.clip();
    ctx.fillStyle=dark; ctx.beginPath(); ctx.arc(ex+lx,ey+ly,6.5,0,Math.PI*2); ctx.fill(); ctx.restore();
    ctx.strokeStyle=dark; ctx.lineWidth=2; ctx.beginPath(); ctx.ellipse(ex,ey,17,11*open,0,Math.PI,Math.PI*2); ctx.stroke();
    // brow
    const inner=g("browInnerUp"), outer=g("browOuterUp"+side), down=g("browDown"+side);
    const by=ey-24; ctx.lineWidth=5; ctx.lineCap="round"; ctx.beginPath();
    ctx.moveTo(ex+s*20, by-outer*12+down*5); ctx.quadraticCurveTo(ex, by-6-outer*6-inner*6+down*7, ex-s*16, by-inner*13+down*11);
    ctx.stroke();
  }
  // nose
  ctx.strokeStyle=skin2; ctx.lineWidth=3; ctx.beginPath(); ctx.moveTo(yaw*14,-12); ctx.quadraticCurveTo(yaw*22+8,22,yaw*16-2,26-avg("noseSneerLeft","noseSneerRight")*5); ctx.stroke();
  // mouth
  const smileL=g("mouthSmileLeft"), smileR=g("mouthSmileRight"), frownL=g("mouthFrownLeft"), frownR=g("mouthFrownRight");
  const pucker=Math.max(g("mouthPucker"),g("mouthFunnel")*0.8), stretch=avg("mouthStretchLeft","mouthStretchRight");
  const close=g("mouthClose"), press=avg("mouthPressLeft","mouthPressRight"), funnel=g("mouthFunnel");
  const halfW=34*(1-0.45*pucker)+stretch*8+ (smileL+smileR)*5;
  const my=62+jaw*10, shift=(g("mouthRight")-g("mouthLeft"))*8 + yaw*8;
  const gap=Math.max(0,(jaw*34+funnel*10+avg("mouthLowerDownLeft","mouthLowerDownRight")*10+avg("mouthUpperUpLeft","mouthUpperUpRight")*7)*(1-0.9*Math.min(1,close*1.5+press)));
  const cyImgL=my-smileR*11+frownR*9, cyImgR=my-smileL*11+frownL*9; // image-left corner = subject's right
  const xl=shift-halfW, xr=shift+halfW, up=my-gap*0.38-3-pucker*3, lo=my+gap*0.62+3+pucker*3;
  ctx.fillStyle=dark; ctx.beginPath(); ctx.moveTo(xl,cyImgL); ctx.quadraticCurveTo(shift,up-gap*0.2,xr,cyImgR); ctx.quadraticCurveTo(shift,lo+gap*0.35,xl,cyImgL); ctx.fill();
  if(gap>7){ctx.fillStyle="#fff";ctx.globalAlpha=.85;ctx.beginPath();ctx.moveTo(xl+9,(cyImgL+up)/2+1);ctx.quadraticCurveTo(shift,up+1,xr-9,(cyImgR+up)/2+1);ctx.quadraticCurveTo(shift,up+Math.min(7,gap*.3),xl+9,(cyImgL+up)/2+1);ctx.fill();ctx.globalAlpha=1;}
  ctx.strokeStyle=lip; ctx.lineWidth=5+pucker*3-press*2; ctx.lineJoin="round";
  ctx.beginPath(); ctx.moveTo(xl,cyImgL); ctx.quadraticCurveTo(shift,up-gap*0.2,xr,cyImgR); ctx.quadraticCurveTo(shift,lo+gap*0.35,xl,cyImgL); ctx.closePath(); ctx.stroke();
  ctx.restore();
}


// ---------------------------------------------------------------------------------------------
// Robot face. Geometry, proportions, glow and mirrored view are the project's original renderer
// (legacy/modules/templates/index.html: mapBlendshapesToRobot + render). Improvements on top:
//   * the mouth is ONE shape (a bendable rounded bar) so it opens from the closed line without a pop,
//   * brows fade in/out instead of popping, bilabial closures (p/b/m) really close the mouth,
//   * real head rotation (yaw/pitch/roll) instead of the face position in the camera image,
//   * a "life" layer mixed with the AI output: breathing, gaze saccades, blinks with squash & stretch,
//     speech-driven emphasis (brows/eyes/glow react to syllable energy), spring dynamics with overshoot.
// ---------------------------------------------------------------------------------------------
function mapRobot(f){
  const v = n => f ? f[CH[n]] : 0;
  const calcEye = side => {
    const blink=v("eyeBlink"+side), wide=v("eyeWide"+side), squint=v("eyeSquint"+side);
    return {h:Math.max(0.05,1.0-blink+wide*0.5-squint*0.6), w:1.0-squint*0.2, squeeze:squint,
            y_offset:(v("eyeLookDown"+side)-v("eyeLookUp"+side))*0.8, x_offset:v("eyeLookOut"+side)-v("eyeLookIn"+side)};
  };
  const open=Math.max(0, v("jawOpen")-v("mouthClose")-(v("mouthPressLeft")+v("mouthPressRight"))*0.25);
  const wide=(v("mouthSmileLeft")+v("mouthSmileRight"))*0.5, narrow=v("mouthPucker")+v("mouthFunnel");
  const frown=(v("mouthFrownLeft")+v("mouthFrownRight"))*0.5;
  return {
    pose:{yaw:v("headYaw"), pitch:v("headPitch"), roll:v("headRoll")},
    eyeL:calcEye("Left"), eyeR:calcEye("Right"),
    browL:{y:v("browInnerUp")-v("browDownLeft"), rot:(v("browInnerUp")*0.5)-(v("browOuterUpLeft")*0.3)-v("browDownLeft")},
    browR:{y:v("browInnerUp")-v("browDownRight"), rot:-(v("browInnerUp")*0.5)+(v("browOuterUpRight")*0.3)+v("browDownRight")},
    mouth:{openY:open, width:1.0+(wide*0.4)-(narrow*0.5)+(v("mouthStretchLeft")+v("mouthStretchRight"))*0.2,
           smile:wide-frown, pucker:v("mouthPucker"), funnel:v("mouthFunnel")}};
}

// Second-order springs: fast & critically damped for the mouth (lip-sync must not lag),
// softer with a little overshoot for eyes, brows and head (that overshoot is what reads as "alive").
function makeSprings(state){ const o={}; for(const k in state) o[k]=(typeof state[k]==="object")?makeSprings(state[k]):{x:state[k],v:0}; return o; }
function stepSprings(sp, target, dt, freq, damp){
  for(const k in target){
    if(typeof target[k]==="object"){ const mouth=(k==="mouth"); stepSprings(sp[k],target[k],dt,mouth?14:freq,mouth?1.0:damp); continue; }
    const s=sp[k], w=2*Math.PI*freq, acc=w*w*(target[k]-s.x)-2*damp*w*s.v;
    s.v+=acc*dt; s.x+=s.v*dt;
  }
}
function readSprings(sp){ const o={}; for(const k in sp) o[k]=("x" in sp[k] && typeof sp[k].x==="number")?sp[k].x:readSprings(sp[k]); return o; }

// Life layer: everything the audio cannot know, generated procedurally and blended with the AI frame.
function makeLife(seed){
  let r=seed*9301+49297; const rnd=()=>{r=(r*233280+12345)%2147483647; return (r%100000)/100000;};
  return {rnd, t:0, gazeX:0, gazeY:0, gazeTX:0, gazeTY:0, nextGaze:1, blink:-1, nextBlink:1.5+rnd()*2, energy:0, emphasis:0, lastOpen:0, phase:rnd()*10};
}
function applyLife(L, target, dt, speaking, hasOwnBlinks){
  L.t+=dt;
  // syllable energy from the AI mouth: drives emphasis (brow flash, eye widening, glow)
  const open=target.mouth.openY, vel=Math.max(0,(open-L.lastOpen)/Math.max(dt,1e-3)); L.lastOpen=open;
  L.energy+=(Math.min(1,open*2.2)-L.energy)*Math.min(1,dt*6);
  L.emphasis=Math.max(L.emphasis*Math.exp(-dt*5), Math.min(1,vel*0.22));
  // gaze: saccades, calmer while speaking (eye contact), wandering when idle
  L.nextGaze-=dt;
  if(L.nextGaze<=0){ const amp=speaking?0.35:1.0; L.gazeTX=(L.rnd()*2-1)*amp; L.gazeTY=(L.rnd()*2-1)*0.5*amp;
    if(L.rnd()<0.45){L.gazeTX=0;L.gazeTY=0;} L.nextGaze=(speaking?1.2:0.7)+L.rnd()*2.2; }
  const gk=Math.min(1,dt*18); L.gazeX+=(L.gazeTX-L.gazeX)*gk; L.gazeY+=(L.gazeTY-L.gazeY)*gk;
  // blinks (skipped when the track already carries blinks), often right after a gaze jump, like people do
  let lid=0;
  if(!hasOwnBlinks){
    L.nextBlink-=dt;
    if(L.blink<0 && L.nextBlink<=0){ L.blink=0; }
    if(L.blink>=0){ L.blink+=dt/0.16; lid=Math.sin(Math.PI*Math.min(1,L.blink)); if(L.blink>=1){L.blink=-1; L.nextBlink=1.8+L.rnd()*3.5; if(L.rnd()<0.2) L.nextBlink=0.25;} }
  }
  const breathe=Math.sin(L.t*1.3+L.phase), sway=Math.sin(L.t*0.53+L.phase*2), sway2=Math.sin(L.t*0.37+L.phase*3);
  const talk=0.4+L.energy*0.9;
  target.pose.yaw+=sway*0.035*talk+L.gazeX*0.05; target.pose.pitch+=sway2*0.03*talk+L.emphasis*0.06+L.gazeY*0.03; target.pose.roll+=Math.sin(L.t*0.71+L.phase)*0.025*talk;
  target.eyeL.x_offset-=L.gazeX; target.eyeR.x_offset+=L.gazeX;  // both eyes look the same way
  for(const e of [target.eyeL,target.eyeR]){ e.y_offset+=L.gazeY*0.8; e.h=Math.max(0.05,e.h*(1-lid)+L.emphasis*0.12); }
  target.browL.y+=L.emphasis*0.35; target.browR.y+=L.emphasis*0.35;
  return {scale:1+breathe*0.008, bob:breathe*0.006, glow:0.75+0.25*L.energy+0.3*L.emphasis};
}

// Perimeter of a rounded rect (w x h, radius r), bent along y = bend * (x / (w/2))^2.
function bentBar(c, w, h, r, bend){
  r=Math.min(r,w/2,h/2); const pts=[], N=7, hw=w/2, hh=h/2;
  const arc=(cx,cy,a0)=>{ for(let i=0;i<=N;i++){ const a=a0+(Math.PI/2)*i/N; pts.push([cx+r*Math.cos(a),cy+r*Math.sin(a)]); } };
  arc(hw-r,-hh+r,-Math.PI/2); arc(hw-r,hh-r,0); arc(-hw+r,hh-r,Math.PI/2); arc(-hw+r,-hh+r,Math.PI);
  const M=10, out=[];  // subdivide the long edges so the bend is smooth
  for(let i=0;i<pts.length;i++){ const p=pts[i], q=pts[(i+1)%pts.length]; for(let k=0;k<M;k++){ out.push([p[0]+(q[0]-p[0])*k/M, p[1]+(q[1]-p[1])*k/M]); } }
  c.beginPath(); out.forEach(([x,y],i)=>{ const yy=y+bend*(x/hw)*(x/hw); i?c.lineTo(x,yy):c.moveTo(x,yy); }); c.closePath();
}

function drawRobot(ctx, s, fx, W, H){
  ctx.clearRect(0,0,W,H); ctx.save();
  const U=H*0.62*fx.scale;
  // mirrored view, as in the original (you look at the robot like in a mirror)
  ctx.translate(W*(0.5-s.pose.yaw*0.12), H*(0.5-s.pose.pitch*0.09+fx.bob)); ctx.rotate(-s.pose.roll*Math.PI/4); ctx.scale(U,U);
  ctx.shadowBlur=30*(W/600)*fx.glow; ctx.shadowColor="#00a2ff"; ctx.fillStyle="#FFFFFF"; ctx.strokeStyle="#FFFFFF";
  const rr=(x,y,w,h,r)=>{ r=Math.min(r,w/2,h/2); ctx.beginPath(); ctx.moveTo(x-w/2+r,y-h/2); ctx.arcTo(x+w/2,y-h/2,x+w/2,y+h/2,r); ctx.arcTo(x+w/2,y+h/2,x-w/2,y+h/2,r); ctx.arcTo(x-w/2,y+h/2,x-w/2,y-h/2,r); ctx.arcTo(x-w/2,y-h/2,x+w/2,y-h/2,r); ctx.closePath(); };
  [s.eyeL,s.eyeR].forEach((e,i)=>{
    const brow=i?s.browR:s.browL;
    ctx.save(); ctx.translate((i?0.22:-0.22)-s.pose.yaw*0.02+e.x_offset*0.02*(i?1:-1), -0.15+e.y_offset*0.1);
    ctx.rotate((i?-1:1)*e.squeeze*0.2+brow.rot*0.5);
    // squash & stretch: a closing eye gets a little wider, like a classic animated blink
    const hN=Math.max(0.04,e.h), w=0.16*e.w*(1+0.22*Math.max(0,1-hN)), h=0.14*hN;
    rr(0,0,w*2,h*2,0.04); ctx.fill();
    const act=Math.max(Math.abs(brow.y),e.squeeze*0.5), alpha=Math.min(1,Math.max(0,(act-0.06)/0.08));
    if(alpha>0){ ctx.globalAlpha=alpha; rr(0,-h-0.05-brow.y*0.05,w*2.2,0.03,0.01); ctx.fill(); ctx.globalAlpha=1; }
    ctx.restore();
  });
  ctx.save(); ctx.translate(-s.pose.yaw*0.015,0.2);
  const m=s.mouth, cw=0.35*m.width*(1-m.pucker*0.6), ch=Math.max(0.015,0.25*m.openY+m.pucker*0.1+m.funnel*0.04), sc=-m.smile*0.1;
  const round=Math.min(1,Math.max(0,(m.pucker+m.funnel*0.6-0.25)/0.4)), radius=0.05+(Math.min(cw,ch)-0.05)*round;
  const openness=Math.min(1,ch/0.06);  // closed: bends like the original smile line; open: mostly shifts, like the original
  ctx.translate(0,sc*0.5*openness);
  bentBar(ctx, cw*2, ch*2, Math.max(radius,Math.min(ch,0.05)), sc*(1-0.55*openness));
  ctx.fill();
  ctx.restore(); ctx.restore();
}

function selectClip(i){
  clip = DATA.clips[i];
  document.querySelectorAll("#clips button").forEach((b,k)=>b.setAttribute("aria-pressed",k===i));
  const audio=document.getElementById("audio"); audio.src=clip.audio;
  const faces=document.getElementById("faces"); faces.innerHTML=""; views=[];
  for(const [name,frames] of Object.entries(clip.tracks)){
    const card=document.createElement("div"); card.className="card";
    card.innerHTML=`<h2>${name}</h2><canvas width="600" height="640"></canvas><div class="bars">`+BARS.map(b=>`<span>${b}</span><div class="bar"><i></i></div>`).join("")+`</div>`;
    faces.appendChild(card);
    views.push({frames,ctx:card.querySelector("canvas").getContext("2d"),bars:[...card.querySelectorAll(".bar i")],springs:makeSprings(mapRobot(null)),life:makeLife(views.length+1),
      alive:name!==TRACKER, ownBlinks:name===ALIVE});
  }
}
let lastNow=performance.now();
function tick(){
  const now=performance.now(), dt=Math.min(0.05,(now-lastNow)/1000)||0.016; lastNow=now;
  if(clip){
    const a=document.getElementById("audio");
    const t=(a.paused && a.currentTime===0) ? PREVIEW_T : a.currentTime;  // #t=5 previews a frame before playback
    for(const v of views){
      const k=Math.min(v.frames.length-1,Math.max(0,Math.round(t*DATA.fps))), f=v.frames[k];
      if(MODE==="robot"){
        const target=mapRobot(f), speaking=!a.paused;
        const fx=(v.alive&&LIFE)?applyLife(v.life,target,dt,speaking,v.ownBlinks):{scale:1,bob:0,glow:0.8};
        for(let k=0;k<2;k++) stepSprings(v.springs,target,dt/2,(v.alive&&LIFE)?5.5:9,(v.alive&&LIFE)?0.62:1.0);
        drawRobot(v.ctx,readSprings(v.springs),fx,600,640); v.ctx.canvas.className="robot"; }
      else { drawFace(v.ctx,f,600,640); v.ctx.canvas.className=""; }
      BARS.forEach((b,j)=>{const x=f[CH[b]]; v.bars[j].style.width=(b.startsWith("head")?Math.abs(x):x)*100+"%";});
    }
  }
  requestAnimationFrame(tick);
}
document.getElementById("sub").textContent=`${DATA.model} · style « ${DATA.style} » · ${DATA.fps} fps · ${DATA.clips.length} extrait(s)`;
DATA.clips.forEach((c,i)=>{const b=document.createElement("button");b.textContent=c.title;b.onclick=()=>selectClip(i);document.getElementById("clips").appendChild(b);});
for(const m of ["robot","human"]) document.getElementById("m-"+m).onclick=()=>{MODE=m;
  document.getElementById("m-robot").setAttribute("aria-pressed",m==="robot"); document.getElementById("m-human").setAttribute("aria-pressed",m==="human");};
document.getElementById("m-life").onclick=e=>{LIFE=!LIFE; e.target.setAttribute("aria-pressed",LIFE);};
selectClip(0); tick();
</script></body></html>
"""
