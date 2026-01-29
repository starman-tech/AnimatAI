# File: Projet_IA/README.md

# Projet_IA: Système d'IA pour Robot avec Blendshapes

Ce projet est une restructuration propre du code original fourni, adapté pour une exécution locale sur votre ordinateur (sans Google Colab). Il conserve le fonctionnement principal des composants originaux :
- **Entraînement** : Entraîne un modèle IA sur des données locales (basé sur la CELLULE 4).
- **Mining** : Extrait des données de vidéos YouTube (basé sur la CELLULE 2.5).
- **Site Web** : Lance un serveur web avec interface pour tester le robot (basé sur la CELLULE 2).

Aucun changement majeur n'a été apporté au cœur des algorithmes ; seuls les chemins ont été adaptés pour un environnement local (pas de Google Drive par défaut, mais configurable via variables d'environnement). Le projet est contrôlable via :
- **Terminal** : Via `main.py` avec des arguments CLI.
- **Dashboard HTML** : Un mini tableau de bord web (Flask) pour lancer les tâches via une interface graphique.

## Prérequis
- Python 3.8+ installé.
- Un token ngrok (gratuit) pour exposer le site web publiquement si nécessaire. Ajoutez-le via `ngrok config add-authtoken VOTRE_TOKEN` ou dans le code de `website.py`.
- Modèles Vosk et Mediapipe seront téléchargés automatiquement lors de la première exécution.
- Pour le mining : Accès internet pour télécharger des vidéos YouTube.

## Installation
1. Clonez ou téléchargez le projet dans un dossier.
2. Installez les dépendances :
pip install -r requirements.txt
textOu utilisez l'option "Installer" dans le dashboard/terminal.

3. Configurez les chemins si nécessaire :
- Dans `modules/training.py` et `modules/mining.py`, `BASE_DIR` est défini comme le répertoire courant par défaut. Vous pouvez le changer pour un dossier spécifique (ex: pour simuler un "Drive").

4. Lancez le projet :
python main.py
text- Cela ouvrira le dashboard à http://127.0.0.1:5001 (par défaut).
- Ou utilisez des arguments CLI (voir ci-dessous).

## Utilisation via Terminal (CLI)
Exécutez `python main.py [commande]` :
- `install` : Installe les dépendances via `requirements.txt`.
- `train` : Lance l'entraînement du modèle.
- `mine` : Lance le mining de données YouTube (interactif).
- `website` : Lance le serveur web pour le site.
- `dashboard` : Lance explicitement le dashboard web (défaut si pas d'argument).

Exemple :
python main.py train
text## Utilisation via Dashboard
1. Lancez `python main.py`.
2. Ouvrez http://127.0.0.1:5001 dans votre navigateur.
3. Le dashboard propose des boutons pour :
   - Installer les dépendances.
   - Lancer l'entraînement.
   - Lancer le mining (ouvrira un menu interactif dans le terminal, car il est CLI-heavy).
   - Lancer le site web (exposera via ngrok si configuré).
   - Voir les logs/status (zone de texte mise à jour).

Note : Les tâches longues (comme l'entraînement ou mining) s'exécutent en arrière-plan via threads, avec logs affichés dans le dashboard.

## Structure du Projet
- `modules/` : Contient les codes originaux adaptés.
  - `training.py` : Code d'entraînement.
  - `mining.py` : Code de mining YouTube.
  - `website.py` : Code du serveur web.
- `dashboard/` : Mini dashboard Flask.
  - `app.py` : Serveur du dashboard.
  - `templates/index.html` : Interface HTML.
- `utils/` : Outils helpers.
  - `installer.py` : Gestion des installations.
- `main.py` : Point d'entrée unique (CLI + Dashboard).
- `requirements.txt` : Liste des libs.
- `README.md` : Ce fichier.

## Notes Importantes
- **Chemins Locaux** : Tous les dossiers (`saved_data_58`, `models_58`, etc.) seront créés localement dans le répertoire du projet. Pas de montage Drive.
- **Ngrok** : Pour le site web, configurez votre token ngrok. Sans cela, le site sera local seulement.
- **Performances** : L'entraînement et mining utilisent GPU si disponible (via Torch/CUDA).
- **Erreurs** : Si des modèles (Vosk/Mediapipe) manquent, ils seront téléchargés auto. Vérifiez les logs.
- **Améliorations** : Le dashboard est basique ; il lance les modules via subprocess pour éviter les blocages.

Si des problèmes, vérifiez les logs dans le terminal ou dashboard.# AnimatAI
