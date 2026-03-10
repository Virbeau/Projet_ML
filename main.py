import json
from datetime import datetime
import numpy as np
from multiprocessing import Pool, cpu_count

# 1. On importe tes deux modules !
from generate_mesh1 import generate_mesh_instance
from solver import solve_instance 

def process_single_instance(args):
    """
    Fonction worker pour générer une instance en parallèle.
    Prend un tuple (index, m, n, H, B, iters) et retourne le record complet.
    """
    i, m, n, H, B, iters = args
    
    # ÉTAPE A : Génération de la topologie
    G, record = generate_mesh_instance(m=m, n=n, seed=i)
    
    # Préparation des paramètres pour le solveur
    terminals = record["terminals"]
    repairable_nodes = record["repairable_nodes"]
    p_fail_repairable = np.array([record["features"][v]["p_fail"] for v in repairable_nodes])
    c_cost_repairable = np.array([record["features"][v]["c_cost"] for v in repairable_nodes])
    
    params = {
        "p_fail": p_fail_repairable,
        "c_cost": c_cost_repairable,
        "repairable_nodes": repairable_nodes
    }
    
    # ÉTAPE B : Calcul de la solution optimale
    pi_star, J_star, _, _ = solve_instance(
        G=G, 
        terminals=terminals, 
        criterion="terminal_connectivity", 
        params=params, 
        H=H, 
        B=B, 
        iters=iters
    )
    
    # ÉTAPE C : Ajout de la solution au record
    record["H"] = H
    record["B"] = B
    record["pi_star"] = pi_star.tolist()
    record["J_star"] = float(J_star)
    
    return record


def main():
    
    # Configuration
    n_instances = 5
    m, n = 2, 3  # Option A : Graphe réduit (6 nœuds → 2^4=16 états au lieu de 1024)
    H = 10
    B = 2.0
    iters = 50
    
    Parallélisation
    n_workers = min(cpu_count(), n_instances)  # Utilise tous les CPU disponibles
    
    
    # Préparation des tâches
    tasks = [(i, m, n, H, B, iters) for i in range(n_instances)]
    
    # Exécution parallèle
    with Pool(processes=n_workers) as pool:
        dataset = pool.map(process_single_instance, tasks)
        
    # ÉTAPE D : Sauvegarde finale
    out_path = "dataset_mesh_test.json"
    with open(out_path, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(), 
            "instances": dataset
        }, f, indent=2)
        
    print(f"Terminé ! {n_instances} instances sauvegardées dans {out_path}")

if __name__ == "__main__":
    main()