import json
from datetime import datetime
import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count
import random
import time
import sys
import os

# Tentative d'import de tqdm, fallback sinon
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm non disponible. Installez-le avec: pip install tqdm")

# 1. On importe les trois générateurs !
from generate_mesh1 import generate_mesh_instance
from generate_sp1 import generate_sp_instance
from generate_er import generate_er_instance
from solver import solve_instance 


def sample_crisis_multiplier():
    """
    Tire alpha selon la règle C:
      - 80%: alpha in [0.6, 1.4]
      - 10%: alpha in [0.1, 0.4] (famine)
      - 10%: alpha in [2.0, 3.0] (abondance)
    Retourne (alpha, regime)
    """
    r = random.random()
    if r < 0.80:
        return random.uniform(0.6, 1.4), "normal"
    if r < 0.90:
        return random.uniform(0.1, 0.4), "famine"
    return random.uniform(2.0, 3.0), "abondance"


def compute_anchor_budget(G, source, target):
    """
    Règle B: Bref = L * 5.5, où L est la longueur du plus court chemin S->T.
    """
    L = nx.shortest_path_length(G, source=source, target=target)
    B_ref = L * 5.5
    return L, B_ref

def process_single_instance(args):
    """
    Fonction worker pour générer une instance en parallèle.
    Prend un tuple (index, graph_type, params, H, iters) et retourne le record complet.
    graph_type: "mesh", "sp", ou "er"
    params: pour mesh = (m, n), pour sp = (num_repairable,), pour er = (num_nodes, p)
    """
    i, graph_type, params, H, iters = args
    
    # ÉTAPE A : Génération de la topologie selon le type
    if graph_type == "mesh":
        m, n = params
        G, record = generate_mesh_instance(m=m, n=n, seed=i)
        graph_label = f"{m}x{n}"
    elif graph_type == "sp":
        num_repairable = params[0]
        G, record = generate_sp_instance(num_repairable=num_repairable, seed=i)
        graph_label = f"sp{num_repairable}"
    else:  # graph_type == "er"
        num_nodes, p = params
        G, record = generate_er_instance(num_nodes=num_nodes, p=p, seed=i)
        graph_label = f"n{num_nodes}_p{p}"
    
    # Préparation des paramètres pour le solveur (identique pour les deux types)
    terminals = record["terminals"]
    source, target = terminals[0], terminals[1]

    # ===== Règles B & C : Budget ancré sur le plus court chemin + multiplicateur de crise =====
    L, B_ref = compute_anchor_budget(G, source, target)
    alpha, alpha_regime = sample_crisis_multiplier()
    B = round(alpha * B_ref, 2)

    repairable_nodes = record["repairable_nodes"]
    p_fail_repairable = np.array([record["features"][v]["p_fail"] for v in repairable_nodes])
    c_cost_repairable = np.array([record["features"][v]["c_cost"] for v in repairable_nodes])
    
    params_solver = {
        "p_fail": p_fail_repairable,
        "c_cost": c_cost_repairable,
        "repairable_nodes": repairable_nodes
    }
    
    # ÉTAPE B : Calcul de la solution optimale
    pi_star, J_star, _, _ = solve_instance(
        G=G, 
        terminals=terminals, 
        criterion="terminal_connectivity", 
        params=params_solver, 
        H=H, 
        B=B, 
        iters=iters
    )
    
    # ÉTAPE C : Conversion au format GNN (PyTorch Geometric)
    all_nodes = sorted(G.nodes())  # Ordre déterministe
    n_nodes = len(all_nodes)
    
    # Construction de la matrice de features X (n_nodes × 9)
    # Features: [p_fail, c_cost, is_source, is_target, in_degree, out_degree, distance_to_target, B, H]
    x = []
    for node in all_nodes:
        feat = record["features"][node]
        x.append([
            feat["p_fail"],
            feat["c_cost"],
            float(feat["is_source"]),
            float(feat["is_target"]),
            float(feat["in_degree"]),
            float(feat["out_degree"]),
            float(feat["distance_to_target"]),
            B,  # Budget global (répété pour chaque nœud)
            float(H)  # Horizon global (répété pour chaque nœud)
        ])
    
    # Construction du vecteur de labels Y (n_nodes)
    # pi_star contient les valeurs pour les nœuds réparables uniquement
    # On met 0.0 pour les terminaux (non réparables)
    y = []
    pi_dict = {repairable_nodes[i]: float(pi_star[i]) for i in range(len(pi_star))}
    for node in all_nodes:
        if node in terminals:
            y.append(0.0)  # Terminaux : pas de réparation allouée
        else:
            y.append(pi_dict[node])
    
    # Format GNN-ready
    gnn_record = {
        "topology_type": f"{graph_type}_{graph_label}",
        "B": float(B),
        "B_ref": float(B_ref),
        "alpha": float(alpha),
        "alpha_regime": alpha_regime,
        "shortest_path_length": int(L),
        "H": int(H),
        "graph": {
            "nodes": all_nodes,
            "edges": list(G.edges()),
            "is_directed": True
        },
        "x": x,  # Matrice de features (n_nodes × 9)
        "y": y,  # Labels (n_nodes)
        "J_star": float(J_star),
        
        # Métadonnées additionnelles (optionnel, pour debug)
        "terminals": terminals,
        "repairable_nodes": repairable_nodes,
        "n_nodes": n_nodes,
        "n_edges": len(G.edges())
    }
    
    return gnn_record


def main():
    # ========== CONFIGURATION ==========
    n_instances = 1000

    # ===== PLAGES POUR H (B est désormais calculé via règles B/C) =====
    H_default_min, H_default_max = 5, 20      # SP / ER
    H_mesh_min, H_mesh_max = 15, 25           # Spécifique Mesh
    
    # ===== CONFIGURATIONS MESH (1/3 du dataset) =====
    # Variété de 4 à 12 nœuds (2 à 10 réparables)
    mesh_configs = [
        ("mesh", (2, 2), 20),  # 4 noeuds,   2 réparables, 4 états     - RAPIDE
        ("mesh", (2, 3), 25),  # 6 noeuds,   4 réparables, 16 états    - RAPIDE
        ("mesh", (3, 2), 25),  # 6 noeuds,   4 réparables, 16 états    - RAPIDE
        ("mesh", (3, 3), 35),  # 9 noeuds,   7 réparables, 128 états   - MOYEN
        ("mesh", (2, 4), 30),  # 8 noeuds,   6 réparables, 64 états    - MOYEN
        ("mesh", (2, 5), 35),  # 10 noeuds,  8 réparables, 256 états   - MOYEN
        ("mesh", (3, 4), 50),  # 12 noeuds, 10 réparables, 1024 états  - LENT
        ("mesh", (4, 3), 50),  # 12 noeuds, 10 réparables, 1024 états  - LENT
    ]
    # Distribution pondérée : plus de petits (rapides), moins de gros (lents)
    mesh_weights = [0.10, 0.10, 0.15, 0.20, 0.20, 0.13, 0.09, 0.03]  # Somme = 1.0
    
    # ===== CONFIGURATIONS SÉRIE-PARALLÈLE (1/3 du dataset) =====
    # Variété de 4 à 12 nœuds (2 à 10 réparables)
    sp_configs = [
        ("sp", (2,), 20),   # 4 noeuds total,   2 réparables, 4 états     - RAPIDE
        ("sp", (3,), 20),   # 5 noeuds total,   3 réparables, 8 états     - RAPIDE
        ("sp", (4,), 25),   # 6 noeuds total,   4 réparables, 16 états    - RAPIDE
        ("sp", (5,), 30),   # 7 noeuds total,   5 réparables, 32 états    - RAPIDE
        ("sp", (6,), 30),   # 8 noeuds total,   6 réparables, 64 états    - MOYEN
        ("sp", (7,), 35),   # 9 noeuds total,   7 réparables, 128 états   - MOYEN
        ("sp", (8,), 40),   # 10 noeuds total,  8 réparables, 256 états   - MOYEN
        ("sp", (9,), 45),   # 11 noeuds total,  9 réparables, 512 états   - LENT
        ("sp", (10,), 50),  # 12 noeuds total, 10 réparables, 1024 états  - LENT
    ]
    # Distribution pondérée : plus de petits (rapides), moins de gros (lents)
    sp_weights = [0.12, 0.12, 0.15, 0.18, 0.18, 0.12, 0.08, 0.03, 0.02]  # Somme = 1.0
    
    # ===== CONFIGURATIONS ERDŐS-RÉNYI (1/3 du dataset) =====
    # Variété de 4 à 12 nœuds avec différentes densités p
    er_configs = [
        ("er", (4,), 20),
        ("er", (5,), 20),
        ("er", (6,), 25),
        ("er", (7,), 30),
        ("er", (8,), 30),
        ("er", (9,), 35),
        ("er", (10,), 40),
        ("er", (11,), 45),
        ("er", (12,), 50),
    ]
    # Distribution pondérée : plus de petits (rapides), moins de gros (lents)
    er_weights = [0.12, 0.12, 0.15, 0.18, 0.18, 0.12, 0.08, 0.03, 0.02]  # Somme = 1.0
    
    # Configuration simplifiée : répartition 1/3 pour chaque type
    n_mesh = n_instances // 3
    n_sp = n_instances // 3
    n_er = n_instances - n_mesh - n_sp
    
    print("\nGÉNÉRATION DATASET HYBRIDE - RÈGLES B/C + AJUSTEMENTS TOPOLOGIQUES")
    print(
        f"Configuration : {n_instances} instances (1/3 MESH + 1/3 SP + 1/3 ER) | "
        f"Tailles: 4-12 nœuds | H_mesh: [{H_mesh_min},{H_mesh_max}] | H_sp/er: [{H_default_min},{H_default_max}]"
    )
    
    # ========== GÉNÉRATION DES TÂCHES ==========
    random.seed(42)
    tasks = []
    
    # Génération des tâches MESH (première tercio)
    for i in range(n_mesh):
        graph_type, params, iters = random.choices(mesh_configs, weights=mesh_weights)[0]
        H = random.randint(H_mesh_min, H_mesh_max)
        tasks.append((i, graph_type, params, H, iters))
    
    # Génération des tâches SP (deuxième tercio)
    for i in range(n_mesh, n_mesh + n_sp):
        graph_type, params, iters = random.choices(sp_configs, weights=sp_weights)[0]
        H = random.randint(H_default_min, H_default_max)
        tasks.append((i, graph_type, params, H, iters))
    
    # Génération des tâches ER (troisième tercio)
    for i in range(n_mesh + n_sp, n_instances):
        graph_type, params, iters = random.choices(er_configs, weights=er_weights)[0]
        num_nodes = params[0]
        p = round(random.uniform(0.30, 0.45), 2)
        H = random.randint(H_default_min, H_default_max)
        tasks.append((i, graph_type, (num_nodes, p), H, iters))
    
    random.shuffle(tasks)
    
    # ========== PARALLÉLISATION AVEC SUIVI ==========
    n_workers = min(3, cpu_count(), n_instances)
    log_file = "generation_progress.log"
    
    # Ouverture du fichier log
    with open(log_file, "w") as log:
        log.write(f"Démarrage génération : {datetime.now().isoformat()}\n")
        log.write(f"Instances : {n_instances} (1/3 MESH + 1/3 SP + 1/3 ER)\n")
        log.write("Règle B: B_ref = L * 5.5\n")
        log.write("Règle C: alpha 80%[0.6,1.4], 10%[0.1,0.4], 10%[2.0,3.0]\n")
        log.write(f"H Mesh: [{H_mesh_min},{H_mesh_max}] | H SP/ER: [{H_default_min},{H_default_max}]\n")
        log.write("ER p: [0.30,0.45]\n")
        log.write(f"Workers : {n_workers}\n")
        log.write("="*70 + "\n\n")
    
    print(f"\n⚙️  Génération en cours ({n_workers} workers)...")
    print(f"📊 Suivi disponible dans : {log_file}\n")
    
    # Barre de progression
    start_time = time.time()
    
    if HAS_TQDM:
        with Pool(processes=n_workers) as pool:
            dataset = list(tqdm(
                pool.imap_unordered(process_single_instance, tasks),
                total=n_instances,
                unit="graph",
                desc="Génération",
                ncols=80,
                position=0,
                leave=True
            ))
    else:
        # Fallback sans tqdm
        with Pool(processes=n_workers) as pool:
            dataset = pool.map(process_single_instance, tasks)
            print(f"Traitement : {len(dataset)}/{n_instances} graphes générés")
    
    elapsed_time = time.time() - start_time
    
    # Statistiques de génération
    with open(log_file, "a") as log:
        log.write(f"\nGénération terminée : {datetime.now().isoformat()}\n")
        log.write(f"Temps écoulé : {elapsed_time//3600:.0f}h {(elapsed_time%3600)//60:.0f}m {elapsed_time%60:.1f}s\n")
        log.write(f"Temps moyen/graphe : {elapsed_time/n_instances:.2f}s\n")
        log.write(f"Graphes/heure : {n_instances/(elapsed_time/3600):.0f}\n")
    
    # ========== SAUVEGARDE ==========
    out_path = "dataset_hybrid_mesh_sp_er_v2_1000.json"
    
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "n_instances": n_instances,
                "n_mesh": n_mesh,
                "n_sp": n_sp,
                "n_er": n_er,
                "H_range_mesh": {"min": H_mesh_min, "max": H_mesh_max},
                "H_range_sp_er": {"min": H_default_min, "max": H_default_max},
                "budget_rule": {
                    "anchor": "B_ref = shortest_path_length * 5.5",
                    "alpha_distribution": {
                        "normal": {"probability": 0.80, "range": [0.6, 1.4]},
                        "famine": {"probability": 0.10, "range": [0.1, 0.4]},
                        "abondance": {"probability": 0.10, "range": [2.0, 3.0]}
                    }
                },
                "er_p_range": {"min": 0.30, "max": 0.45},
                "mesh_configs": [{"type": gt, "params": p, "iters": it, "weight": w} 
                                  for (gt, p, it), w in zip(mesh_configs, mesh_weights)],
                "sp_configs": [{"type": gt, "params": p, "iters": it, "weight": w} 
                                for (gt, p, it), w in zip(sp_configs, sp_weights)],
                "er_configs": [{"type": gt, "params": p, "iters": it, "weight": w} 
                                for (gt, p, it), w in zip(er_configs, er_weights)]
            },
            "instances": dataset
        }, f, indent=2)
    
    # ========== STATISTIQUES FINALES ==========
    H_values = [r["H"] for r in dataset]
    B_values = [r["B"] for r in dataset]
    alpha_values = [r["alpha"] for r in dataset]
    L_values = [r["shortest_path_length"] for r in dataset]
    j_values = [r["J_star"] for r in dataset]
    n_nodes_values = [r["n_nodes"] for r in dataset]
    regime_counts = {"normal": 0, "famine": 0, "abondance": 0}
    for r in dataset:
        regime_counts[r["alpha_regime"]] = regime_counts.get(r["alpha_regime"], 0) + 1
    
    # Comptage par type de topologie
    topology_counts = {}
    for r in dataset:
        topo_type = r["topology_type"].split("_")[0]
        topology_counts[topo_type] = topology_counts.get(topo_type, 0) + 1
    
    # Affichage résumé
    print("="*70)
    print("✅ GÉNÉRATION COMPLÈTE")
    print("="*70)
    print(f"\n📊 Instances : {n_instances}")
    print(f"   MESH: {topology_counts.get('mesh', 0):,} | SP: {topology_counts.get('sp', 0):,} | ER: {topology_counts.get('er', 0):,}")
    print(f"\n⏱️  Temps total : {elapsed_time//3600:.0f}h {(elapsed_time%3600)//60:.0f}m {elapsed_time%60:.1f}s")
    print(f"   Temps moyen/graphe : {elapsed_time/n_instances:.3f}s")
    print(f"   Débit : {n_instances/(elapsed_time/3600):.0f} graphes/heure")
    print(f"\n📁 Fichier : {out_path}")
    if os.path.exists(out_path):
        print(f"   Taille : {os.path.getsize(out_path) / (1024*1024):.1f} MB")
    print(f"\n📈 Statistiques générées :")
    print(f"   • H (Horizon) : {min(H_values)}-{max(H_values)} (moy: {np.mean(H_values):.1f})")
    print(f"   • L (Plus court chemin) : {min(L_values)}-{max(L_values)} (moy: {np.mean(L_values):.2f})")
    print(f"   • α (Crise) : {min(alpha_values):.2f}-{max(alpha_values):.2f} (moy: {np.mean(alpha_values):.2f})")
    print(
        f"     Régimes α -> normal: {regime_counts.get('normal', 0)} | "
        f"famine: {regime_counts.get('famine', 0)} | abondance: {regime_counts.get('abondance', 0)}"
    )
    print(f"   • B (Budget final)  : {min(B_values):.2f}-{max(B_values):.2f} (moy: {np.mean(B_values):.2f})")
    print(f"   • J* (Fiabilité) : {min(j_values):.4f}-{max(j_values):.4f} (moy: {np.mean(j_values):.4f}±{np.std(j_values):.4f})")
    print(f"   • Nœuds/graphe : {min(n_nodes_values)}-{max(n_nodes_values)} (moy: {np.mean(n_nodes_values):.1f})")
    print("\n" + "="*70)
    
    # Mise à jour du log final
    with open(log_file, "a") as log:
        log.write(f"\nRÉSULTATS FINAUX:\n")
        log.write(f"  - Instances générées: {n_instances}\n")
        log.write(f"  - MESH: {topology_counts.get('mesh', 0)} | SP: {topology_counts.get('sp', 0)} | ER: {topology_counts.get('er', 0)}\n")
        log.write(f"  - Fichier: {out_path}\n")
        log.write(f"  - H: {min(H_values)}-{max(H_values)}\n")
        log.write(f"  - L: {min(L_values)}-{max(L_values)}\n")
        log.write(f"  - alpha: {min(alpha_values):.2f}-{max(alpha_values):.2f}\n")
        log.write(
            f"  - alpha regimes: normal={regime_counts.get('normal', 0)}, "
            f"famine={regime_counts.get('famine', 0)}, abondance={regime_counts.get('abondance', 0)}\n"
        )
        log.write(f"  - B: {min(B_values):.2f}-{max(B_values):.2f}\n")
        log.write(f"  - J*: {np.mean(j_values):.4f}±{np.std(j_values):.4f}\n")


if __name__ == "__main__":
    main()
