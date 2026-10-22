from flask import Flask, render_template, request, redirect, url_for, jsonify
import subprocess
import os
import sys

app = Flask(__name__)

# On définit les chemins par rapport à la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Gestionnaires de processus (globaux pour persister entre les requêtes)
processes = {
    "website": None,
    "mining": None,
    "training": None
}

def get_status():
    """Calcule l'état actuel de chaque module"""
    status = {}
    for k, p in processes.items():
        if p is None:
            status[k] = "STOPPED"
        elif p.poll() is None:
            status[k] = "RUNNING"
        else:
            status[k] = "FINISHED"
    return status

@app.route('/')
def index():
    # 1. Calcul des statuts
    current_status = get_status()
    
    # 2. Compte des fichiers dans /data
    n_files = 0
    if os.path.exists(DATA_DIR):
        n_files = len([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])
    
    # 3. Vérification du modèle
    model_ok = os.path.exists(os.path.join(MODELS_DIR, "face_model.pth"))
    
    # 4. ENVOI DES VARIABLES AU HTML (C'est ici que ça bloquait)
    return render_template('index.html', 
                           status=current_status, 
                           n_files=n_files, 
                           model_ok=model_ok)

@app.route('/action/<action_type>', methods=['POST'])
def action(action_type):
    global processes
    interpreter = sys.executable
    
    if action_type == "start_robot":
        if processes["website"] is None or processes["website"].poll() is not None:
            script = os.path.join(BASE_DIR, "modules", "website.py")
            log = open("website.py.log", "a")
            processes["website"] = subprocess.Popen([interpreter, script], stdout=log, stderr=log)
            
    elif action_type == "stop_robot":
        if processes["website"]:
            processes["website"].terminate()
            processes["website"] = None
            
    elif action_type == "start_mining":
        if processes["mining"] is None or processes["mining"].poll() is not None:
            script = os.path.join(BASE_DIR, "modules", "mining.py")
            log = open("mining.py.log", "a")
            processes["mining"] = subprocess.Popen([interpreter, script], stdout=log)

    elif action_type == "start_training":
        if processes["training"] is None or processes["training"].poll() is not None:
            script = os.path.join(BASE_DIR, "modules", "training.py")
            log = open("training.py.log", "a")
            processes["training"] = subprocess.Popen([interpreter, script], stdout=log)
            
    elif action_type == "add_url":
        url = request.form.get("url")
        if url:
            # Import dynamique pour ne pas bloquer
            sys.path.append(os.path.join(BASE_DIR, "modules"))
            try:
                from mining import add_url
                add_url(url)
            except ImportError:
                pass

    return redirect(url_for('index'))

@app.route('/logs/<script>')
def logs(script):
    try:
        log_path = f"{script}.py.log"
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                # On prend les 50 dernières lignes pour la lisibilité
                lines = f.readlines()
                return "".join(lines[-50:]).replace('\n', '<br>')
        return "Aucun log disponible."
    except Exception as e:
        return f"Erreur de lecture : {str(e)}"

if __name__ == "__main__":
    # On lance sur le port 5001 pour éviter le conflit avec le futur site robot (5000)
    app.run(port=5001, debug=True)