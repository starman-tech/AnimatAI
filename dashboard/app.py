# File: Projet_IA/dashboard/app.py
# Mini dashboard Flask pour contrôler les modules.

from flask import Flask, render_template, request, jsonify
import subprocess
import threading
import os

app = Flask(__name__, template_folder='templates')

# Logs globaux (simples)
logs = []

def run_in_thread(cmd):
    def target():
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.getcwd())
        for line in iter(process.stdout.readline, ''):
            logs.append(line.strip())
        process.stdout.close()
        process.wait()

    thread = threading.Thread(target=target)
    thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run/<action>', methods=['POST'])
def run_action(action):
    if action == 'install':
        run_in_thread(['python', '-m', 'utils.installer'])
    elif action == 'train':
        run_in_thread(['python', '-m', 'modules.training'])
    elif action == 'mine':
        run_in_thread(['python', '-m', 'modules.mining'])
    elif action == 'website':
        run_in_thread(['python', '-m', 'modules.website'])
    else:
        return jsonify({'error': 'Action inconnue'}), 400
    return jsonify({'status': 'Lancé'})

@app.route('/logs', methods=['GET'])
def get_logs():
    return jsonify({'logs': logs})

def run_dashboard():
    app.run(port=5001, debug=False)

if __name__ == "__main__":
    run_dashboard()