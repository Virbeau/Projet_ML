datasetV7_er = "datasetV7_er.json"
datasetV7_mesh = "datasetV7_mesh.json"
datasetV7_sp = "datasetV7_sp.json"
fusionV7 = "fusionV7.json"
def load_instances(path):

# Importation des bibliothèques pour la gestion des fichiers JSON, l'aléatoire et la date
import json
import random
from datetime import datetime

# Chemins des fichiers de datasets à fusionner
datasetV7_er = "datasetV7_er.json"
datasetV7_mesh = "datasetV7_mesh.json"
datasetV7_sp = "datasetV7_sp.json"

# Nom du fichier de sortie fusionné
fusionV7 = "fusionV7.json"

# Fonction utilitaire pour charger les instances depuis un fichier JSON
def load_instances(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "instances" not in payload:
        raise ValueError(f"Le fichier {path} ne contient pas de clé 'instances'.")
    return payload["instances"]


def main():
    # Initialisation du générateur aléatoire pour la reproductibilité
    random.seed(42)

    # Chargement des instances depuis chaque dataset source
    er_instances = load_instances(datasetV7_er)
    mesh_instances = load_instances(datasetV7_mesh)
    sp_instances = load_instances(datasetV7_sp)

    # Fusion des instances et mélange aléatoire
    merged_instances = er_instances + mesh_instances + sp_instances
    random.shuffle(merged_instances)

    # Construction du dictionnaire de sortie avec métadonnées
    merged_payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source_files": [datasetV7_er, datasetV7_mesh, datasetV7_sp],
            "n_instances": len(merged_instances),
            "shuffle_seed": 42,
            "description": "Fusion des trois datasets V7 train (ER + Mesh + SP)"
        },
        "instances": merged_instances
    }

    # Sauvegarde du dataset fusionné dans un fichier JSON
    with open(fusionV7, "w", encoding="utf-8") as f:
        json.dump(merged_payload, f, indent=2)

    print(f"Fusion terminée: {fusionV7} | {len(merged_instances)} instances")


# Point d'entrée du script
if __name__ == "__main__":
    main()