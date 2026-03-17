import json
import random
from datetime import datetime


datasetV7_er = "datasetV7_er.json"
datasetV7_mesh = "datasetV7_mesh.json"
datasetV7_sp = "datasetV7_sp.json"

fusionV7 = "fusionV7.json"


def load_instances(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "instances" not in payload:
        raise ValueError(f"Le fichier {path} ne contient pas de cle 'instances'.")
    return payload["instances"]


def main():
    random.seed(42)

    er_instances = load_instances(datasetV7_er)
    mesh_instances = load_instances(datasetV7_mesh)
    sp_instances = load_instances(datasetV7_sp)

    merged_instances = er_instances + mesh_instances + sp_instances
    random.shuffle(merged_instances)

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

    with open(fusionV7, "w", encoding="utf-8") as f:
        json.dump(merged_payload, f, indent=2)

    print(f"Fusion terminee: {fusionV7} | {len(merged_instances)} instances")


if __name__ == "__main__":
    main()