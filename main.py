# main.py (updated)
import sys
import subprocess
import threading

def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in iter(process.stdout.readline, ''):
        print(line.strip())
    process.stdout.close()
    process.wait()

def interactive_menu():
    while True:
        print("\n=== Menu Projet IA (Terminal) ===")
        print("1. Install Dependencies")
        print("2. Train Model")
        print("3. Mine Data (YouTube)")
        print("4. Run Website")
        print("5. Run Dashboard")  # New option
        print("6. Quit")
        choice = input("Enter choice (1-6): ")
        
        if choice == '1':
            from utils.installer import install_dependencies
            install_dependencies()
        elif choice == '2':
            from modules.training import train
            train()
        elif choice == '3':
            from modules.mining import run_miner_menu
            run_miner_menu()
        elif choice == '4':
            from modules.website import run_website
            run_website()
        elif choice == '5':  # New: Run dashboard
            from dashboard.app import run_dashboard
            run_dashboard()
        elif choice == '6':
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "install":
            from utils.installer import install_dependencies
            install_dependencies()
        elif command == "train":
            from modules.training import train
            train()
        elif command == "mine":
            from modules.mining import run_miner_menu
            run_miner_menu()
        elif command == "website":
            from modules.website import run_website
            run_website()
        elif command == "dashboard":  # New CLI arg
            from dashboard.app import run_dashboard
            run_dashboard()
        else:
            print("Commande inconnue. Utilisez: install, train, mine, website, dashboard")
            print("Ou lancez sans argument pour le menu interactif.")
    else:
        interactive_menu()