import json
from datetime import datetime
import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count
import random
import time
import os
import argparse

# Tentative d'import de tqdm, fallback sinon
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm non disponible. Installez-le avec: pip install tqdm")

from generate_mesh1 import generate_mesh_instance
from generate_sp1 import generate_sp_instance
from generate_er import generate_er_instance
from solver import solve_instance 


def get_budget_alpha(graph_type):
    """
    Tire le multiplicateur alpha avec une Loi Normale adaptée à la topologie.
    On centre la gaussienne autour du coût du chemin critique pour cibler J* ~ 0.5.
    """
    if graph_type == "mesh":
        # Les mesh sont redondants. On est plus sévère sur le budget.
        mu, sigma = 0.8, 0.2
    elif graph_type == "sp":
        # Les SP ont des goulots d'étranglement, ils ont besoin d'un peu plus.
        mu, sigma = 1.1, 0.3
    else:
        # Les ER sont chaotiques et souvent fragiles.
        mu, sigma = 1.2, 0.3
        
    alpha = np.random.normal(loc=mu, scale=sigma)
    # On évite les extrêmes absurdes
    return float(np.clip(alpha, 0.2, 2.0))


def process_single_instance(args):
    """
    Fonction worker avec Filtre de Jouabilité.
    Boucle jusqu'à trouver un graphe "intéressant" (où 0.5 est atteignable).
    """
    base_seed, graph_type, params, H, iters = args
    attempt = 0
    MAX_ATTEMPTS = 200

    G = record = graph_label = None
    C_min_path = alpha = B = J_min = J_max = pi_star = J_star = L = None
    c_cost_repairable = None
    params_solver = None
    repairable_nodes = terminals = source = target = None

    while attempt < MAX_ATTEMPTS:
        current_seed = base_seed + (attempt * 10000)
        attempt += 1
        
        # 1. Génération
        if graph_type == "mesh":
            m, n = params
            G, record = generate_mesh_instance(m=m, n=n, seed=current_seed)
            graph_label = f"{m}x{n}"
        elif graph_type == "sp":
            num_repairable = params[0]
            G, record = generate_sp_instance(num_repairable=num_repairable, seed=current_seed)
            graph_label = f"sp{num_repairable}"
        else:
            num_nodes, p = params
            G, record = generate_er_instance(num_nodes=num_nodes, p=p, seed=current_seed)
            graph_label = f"n{num_nodes}_p{p}"
            
        terminals = record["terminals"]
        source, target = terminals[0], terminals[-1]
        repairable_nodes = record["repairable_nodes"]
        
        p_fail_repairable = np.array([record["features"][v]["p_fail"] for v in repairable_nodes])
        c_cost_repairable = np.array([record["features"][v]["c_cost"] for v in repairable_nodes])
        
        params_solver = {
            "p_fail": p_fail_repairable,
            "c_cost": c_cost_repairable,
            "repairable_nodes": repairable_nodes
        }
        
        C_total = float(sum(c_cost_repairable))
        if C_total <= 0: C_total = 1.0

        # ==========================================================
        # 2. FILTRE DE JOUABILITÉ (Le réseau est-il intéressant ?)
        # ==========================================================
        _, J_min, _, _ = solve_instance(G, terminals, "terminal_connectivity", params_solver, H, C_total, iters=iters)
        if J_min > 0.60:
            continue
            
        _, J_max, _, _ = solve_instance(G, terminals, "terminal_connectivity", params_solver, H, 0.0, iters=iters)
        if J_max < 0.40:
            continue

        # ==========================================================
        # 3. TIRAGE DU BUDGET (Gaussienne Ajustée)
        # ==========================================================
        def node_cost(n):
            return record["features"][n]["c_cost"] if n in repairable_nodes else 0.0
            
        def edge_weight(u, v, d):
            return node_cost(v)
            
        try:
            cheapest_path = nx.shortest_path(G, source=source, target=target, weight=edge_weight)
            C_min_path = sum(node_cost(n) for n in cheapest_path)
        except nx.NetworkXNoPath:
            continue

        if C_min_path <= 0.01: C_min_path = 0.5
        
        alpha = get_budget_alpha(graph_type)
        B = round(alpha * C_min_path, 2)
        
        # 4. Résolution Finale avec le vrai budget B
        pi_star, J_star, _, _ = solve_instance(
            G=G, terminals=terminals, criterion="terminal_connectivity", 
            params=params_solver, H=H, B=B, iters=iters
        )
        
        L = nx.shortest_path_length(G, source=source, target=target)
        break
    else:
        # Fallback MAX_ATTEMPTS atteint : on force avec le dernier graphe généré
        C_min_path = float(np.sum(c_cost_repairable)) * 0.5 if c_cost_repairable is not None else 1.0
        if C_min_path <= 0: C_min_path = 1.0
        alpha = get_budget_alpha(graph_type)
        B = round(alpha * C_min_path, 2)
        pi_star, J_star, _, _ = solve_instance(
            G=G, terminals=terminals, criterion="terminal_connectivity",
            params=params_solver, H=H, B=B, iters=iters
        )
        J_min = J_max = float(J_star)
        L = nx.shortest_path_length(G, source=source, target=target)
        C_total = float(np.sum(c_cost_repairable)) if c_cost_repairable is not None else 1.0

    # 5. Conversion au format GNN
    all_nodes = sorted(G.nodes())
    n_nodes = len(all_nodes)
    
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
            B,  
            float(H) 
        ])
    
    y = []
    pi_dict = {repairable_nodes[idx]: float(pi_star[idx]) for idx in range(len(pi_star))}
    for node in all_nodes:
        y.append(0.0 if node in terminals else pi_dict[node])
        
    gnn_record = {
        "topology_type": f"{graph_type}_{graph_label}",
        "B": float(B),
        "C_min_path": float(C_min_path),
        "C_total": float(C_total),
        "alpha": float(alpha),
        "shortest_path_length": int(L),
        "H": int(H),
        "attempts_needed": attempt, # Info debug
        "J_min": float(J_min),
        "J_max": float(J_max),
        "graph": {
            "nodes": all_nodes,
            "edges": list(G.edges()),
            "is_directed": True
        },
        "x": x,
        "y": y,
        "J_star": float(J_star),
        "terminals": terminals,
        "repairable_nodes": repairable_nodes,
        "n_nodes": n_nodes,
        "n_edges": len(G.edges())
    }
    return gnn_record
        # 5. Conversion au format GNN


def main():
    parser = argparse.ArgumentParser(description="Génération dataset hybride avec Gaussienne & Filtre de Jouabilité")
    parser.add_argument("--n-instances", type=int, default=1000, help="Nombre total d'instances à générer")
    parser.add_argument("--seed", type=int, default=42, help="Seed globale")
    parser.add_argument("--out-path", type=str, default="dataset_hybrid_filtered_v3.json", help="Fichier JSON de sortie")
    parser.add_argument("--log-file", type=str, default="generation_progress.log", help="Log de progression")
    parser.add_argument("--workers", type=int, default=None, help="Nombre de workers")
    args = parser.parse_args()

    n_instances = args.n_instances
    
    # HORIZONS RÉDUITS pour éviter la mort inévitable du réseau
    H_default_min, H_default_max = 5, 12   # SP / ER
    H_mesh_min, H_mesh_max = 5, 12        # Mesh
    
    mesh_configs = [
        ("mesh", (2, 2), 20), ("mesh", (2, 3), 25), ("mesh", (3, 2), 25), 
        ("mesh", (3, 3), 35), ("mesh", (2, 4), 30), ("mesh", (2, 5), 35), 
        ("mesh", (3, 4), 50), ("mesh", (4, 3), 50),
    ]
    mesh_weights = [0.10, 0.10, 0.15, 0.20, 0.20, 0.13, 0.09, 0.03]
    
    sp_configs = [
        ("sp", (2,), 20), ("sp", (3,), 20), ("sp", (4,), 25), 
        ("sp", (5,), 30), ("sp", (6,), 30), ("sp", (7,), 35), 
        ("sp", (8,), 40), ("sp", (9,), 45), ("sp", (10,), 50),
    ]
    sp_weights = [0.12, 0.12, 0.15, 0.18, 0.18, 0.12, 0.08, 0.03, 0.02]
    
    er_configs = [
        ("er", (4,), 20), ("er", (5,), 20), ("er", (6,), 25),
        ("er", (7,), 30), ("er", (8,), 30), ("er", (9,), 35),
        ("er", (10,), 40), ("er", (11,), 45), ("er", (12,), 50),
    ]
    er_weights = [0.12, 0.12, 0.15, 0.18, 0.18, 0.12, 0.08, 0.03, 0.02]
    
    n_mesh = n_instances // 3
    n_sp = n_instances // 3
    n_er = n_instances - n_mesh - n_sp
    
    print("\nGÉNÉRATION DATASET HYBRIDE - GAUSSIENNE ADAPTATIVE & FILTRE")
    
    random.seed(args.seed)
    tasks = []
    
    for i in range(n_mesh):
        graph_type, params, iters = random.choices(mesh_configs, weights=mesh_weights)[0]
        H = random.randint(H_mesh_min, H_mesh_max)
        tasks.append((i, graph_type, params, H, iters))
    
    for i in range(n_mesh, n_mesh + n_sp):
        graph_type, params, iters = random.choices(sp_configs, weights=sp_weights)[0]
        H = random.randint(H_default_min, H_default_max)
        tasks.append((i, graph_type, params, H, iters))
    
    for i in range(n_mesh + n_sp, n_instances):
        graph_type, params, iters = random.choices(er_configs, weights=er_weights)[0]
        p = round(random.uniform(0.30, 0.45), 2)
        H = random.randint(H_default_min, H_default_max)
        tasks.append((i, graph_type, (params[0], p), H, iters))
    
    random.shuffle(tasks)
    
    n_workers = args.workers if args.workers is not None else min(3, cpu_count(), n_instances)
    
    print(f"\n⚙️  Génération en cours ({n_workers} workers)...")
    
    start_time = time.time()
    if HAS_TQDM:
        with Pool(processes=n_workers) as pool:
            dataset = list(tqdm(pool.imap_unordered(process_single_instance, tasks), total=n_instances))
    else:
        with Pool(processes=n_workers) as pool:
            dataset = pool.map(process_single_instance, tasks)
    
    elapsed_time = time.time() - start_time
    
    with open(args.out_path, "w") as f:
        json.dump({
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "n_instances": n_instances,
                "budget_rule": "Alpha Normal Distribution centered on C_min_path (Mesh:0.8, SP:1.1, ER:1.2)",
                "filter_rule": "Rejected if J_min > 0.60 or J_max < 0.40"
            },
            "instances": dataset
        }, f, indent=2)
    
    # ========== STATISTIQUES ==========
    alpha_values = [r["alpha"] for r in dataset]
    j_values = [r["J_star"] for r in dataset]
    attempts = [r["attempts_needed"] for r in dataset]
    
    print("="*70)
    print("✅ GÉNÉRATION COMPLÈTE")
    print(f"⏱️  Temps total : {elapsed_time:.1f}s")
    print(f"\n📈 Statistiques du nouveau Dataset :")
    print(f"   • Rejets moyens par instance : {np.mean(attempts)-1:.1f}")
    print(f"   • α appliqué (Budget/C_min_path) : {min(alpha_values):.2f} à {max(alpha_values):.2f}")
    print(f"   • J* (Probabilité de panne) : {min(j_values):.4f} à {max(j_values):.4f}")
    print(f"   • Moyenne de J* : {np.mean(j_values):.4f} (Devrait être centrée !)")
    print("="*70)

if __name__ == "__main__":
    main()