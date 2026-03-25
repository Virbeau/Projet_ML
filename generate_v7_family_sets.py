
# Importation des bibliothèques pour la gestion des arguments, JSON, aléatoire, temps, parallélisme et calcul numérique
import argparse
import json
import random
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np

# Import d'une fonction utilitaire pour traiter une instance
from main_production import process_single_instance

# Importation optionnelle de tqdm pour la barre de progression
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False



# Configurations de tailles/types de graphes pour chaque famille (Mesh, SP, ER) et leurs poids de sélection
MESH_CONFIGS = [
    ((2, 2), 20),
    ((2, 3), 25),
    ((3, 2), 25),
    ((3, 3), 35),
    ((2, 4), 30),
    ((2, 5), 35),
    ((3, 4), 50),
    ((4, 3), 50),
]
MESH_WEIGHTS = [0.10, 0.10, 0.15, 0.20, 0.20, 0.13, 0.09, 0.03]

SP_CONFIGS = [
    ((2,), 20),
    ((3,), 20),
    ((4,), 25),
    ((5,), 30),
    ((6,), 30),
    ((7,), 35),
    ((8,), 40),
    ((9,), 45),
    ((10,), 50),
]
SP_WEIGHTS = [0.12, 0.12, 0.15, 0.18, 0.18, 0.12, 0.08, 0.03, 0.02]

ER_NODE_CONFIGS = [
    ((4,), 20),
    ((5,), 20),
    ((6,), 25),
    ((7,), 30),
    ((8,), 30),
    ((9,), 35),
    ((10,), 40),
    ((11,), 45),
    ((12,), 50),
]
ER_WEIGHTS = [0.12, 0.12, 0.15, 0.18, 0.18, 0.12, 0.08, 0.03, 0.02]


def _sample_task(family, mode, rng, min_nodes=15, max_nodes=20):
    """
    Génère une configuration de tâche pour une famille de graphes donnée, selon le mode (benchmark ou autre).
    """
    if mode == "benchmark":
        if family == "er":
            num_nodes = rng.randint(min_nodes, max_nodes)
            p = round(rng.uniform(0.18, 0.30), 2)
            return family, (num_nodes, p), 55
        if family == "sp":
            # total nodes = num_repairable + 2
            rep_min = max(1, min_nodes - 2)
            rep_max = max(rep_min, max_nodes - 2)
            num_repairable = rng.randint(rep_min, rep_max)
            return family, (num_repairable,), 55
        mesh_candidates = [(2, 7), (7, 2), (3, 5), (4, 4), (3, 6), (4, 5)]
        feasible = [sh for sh in mesh_candidates if min_nodes <= sh[0] * sh[1] <= max_nodes]
        if not feasible:
            # Prend la forme la plus proche sans dépasser max_nodes si possible
            below = [sh for sh in mesh_candidates if sh[0] * sh[1] <= max_nodes]
            feasible = below if below else [min(mesh_candidates, key=lambda sh: sh[0] * sh[1])]
        mesh_shape = rng.choice(feasible)
        return family, mesh_shape, 55

    if family == "mesh":
        params, iters = rng.choices(MESH_CONFIGS, weights=MESH_WEIGHTS, k=1)[0]
        return family, params, iters
    if family == "sp":
        params, iters = rng.choices(SP_CONFIGS, weights=SP_WEIGHTS, k=1)[0]
        return family, params, iters

    params, iters = rng.choices(ER_NODE_CONFIGS, weights=ER_WEIGHTS, k=1)[0]
    p = round(rng.uniform(0.27, 0.40), 2)
    return family, (params[0], p), iters


def _build_tasks(
    family,
    n_instances,
    seed,
    h,
    mode,
    min_nodes=15,
    max_nodes=20,
    max_repairable=None,
):
    rng = random.Random(seed)
    tasks = []
    for i in range(n_instances):
        graph_type, params, iters = _sample_task(
            family,
            mode,
            rng,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
        )
        base_seed = seed * 100000 + i * 1000 + (0 if family == "mesh" else (1 if family == "sp" else 2))
        if max_repairable is not None:
            tasks.append((base_seed, graph_type, params, h, iters, int(max_repairable)))
        else:
            tasks.append((base_seed, graph_type, params, h, iters))
    return tasks


def _run_tasks(tasks, workers):
    if HAS_TQDM:
        with Pool(processes=workers) as pool:
            return list(tqdm(pool.imap_unordered(process_single_instance, tasks), total=len(tasks)))
    with Pool(processes=workers) as pool:
        return pool.map(process_single_instance, tasks)


def _save_dataset(out_path, family, split_name, h, dataset, started_at, elapsed):
    alpha_values = [r["alpha"] for r in dataset]
    j_values = [r["J_star"] for r in dataset]
    attempts = [r["attempts_needed"] for r in dataset]

    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "family": family,
            "split": split_name,
            "n_instances": len(dataset),
            "H": int(h),
            "budget_rule": "B = alpha * C_total (risk-aware alpha by family)",
            "duration_seconds": float(elapsed),
            "started_at": started_at,
        },
        "instances": dataset,
        "stats": {
            "alpha_min": float(np.min(alpha_values)),
            "alpha_max": float(np.max(alpha_values)),
            "j_mean": float(np.mean(j_values)),
            "j_min": float(np.min(j_values)),
            "j_max": float(np.max(j_values)),
            "retries_mean": float(np.mean(attempts) - 1.0),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def generate_one(
    out_path,
    family,
    n_instances,
    seed,
    h,
    workers,
    mode,
    split_name,
    min_nodes=15,
    max_nodes=20,
    max_repairable=None,
):
    print(f"\n=== {out_path} | family={family} | n={n_instances} | mode={mode} ===")
    started_at = datetime.now().isoformat()
    t0 = time.time()
    tasks = _build_tasks(
        family=family,
        n_instances=n_instances,
        seed=seed,
        h=h,
        mode=mode,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        max_repairable=max_repairable,
    )
    dataset = _run_tasks(tasks, workers=workers)
    elapsed = time.time() - t0
    _save_dataset(out_path, family, split_name, h, dataset, started_at, elapsed)
    print(f"Saved {out_path} in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Generate V7 datasets by family (train/val/benchmark)")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--H", type=int, default=25, help="Horizon H")
    parser.add_argument("--seed", type=int, default=700, help="Base seed")
    parser.add_argument("--only-generalisation", action="store_true", help="Generate only generalisation files")
    parser.add_argument("--generalisation-n", type=int, default=30, help="Instances per family for generalisation")
    parser.add_argument("--generalisation-min-nodes", type=int, default=15, help="Minimum number of nodes for generalisation")
    parser.add_argument("--generalisation-max-nodes", type=int, default=20, help="Maximum number of nodes for generalisation")
    parser.add_argument("--max-repairable", type=int, default=12, help="Maximum repairable nodes for exact solver safety")
    args = parser.parse_args()

    workers = args.workers if args.workers is not None else min(3, cpu_count())
    print(f"Using workers={workers}, H={args.H}, seed={args.seed}")

    if args.only_generalisation:
        safe_max_nodes = min(args.generalisation_max_nodes, args.max_repairable + 2)
        safe_min_nodes = min(args.generalisation_min_nodes, safe_max_nodes)
        if safe_max_nodes < args.generalisation_max_nodes:
            print(
                f"[WARN] Requested max nodes={args.generalisation_max_nodes} is not exact-solver safe. "
                f"Using solver-safe range [{safe_min_nodes}, {safe_max_nodes}] with max_repairable={args.max_repairable}."
            )

        specs = [
            ("generalisationV7_mesh.json", "mesh", args.generalisation_n, "benchmark", "benchmark"),
            ("generalisationV7_sp.json", "sp", args.generalisation_n, "benchmark", "benchmark"),
            ("generalisationV7_er.json", "er", args.generalisation_n, "benchmark", "benchmark"),
        ]
    else:
        specs = [
            ("datasetV7_mesh.json", "mesh", 3000, "train", "train"),
            ("datasetV7_sp.json", "sp", 3000, "train", "train"),
            ("datasetV7_er.json", "er", 3000, "train", "train"),
            ("testsetV7_mesh.json", "mesh", 300, "validation", "validation"),
            ("testsetV7_sp.json", "sp", 300, "validation", "validation"),
            ("testsetV7_er.json", "er", 300, "validation", "validation"),
            ("generalisationV7_mesh.json", "mesh", 5, "benchmark", "benchmark"),
            ("generalisationV7_sp.json", "sp", 5, "benchmark", "benchmark"),
            ("generalisationV7_er.json", "er", 5, "benchmark", "benchmark"),
        ]

    for idx, (out_path, family, n_instances, mode, split_name) in enumerate(specs):
        local_min_nodes = args.generalisation_min_nodes
        local_max_nodes = args.generalisation_max_nodes
        if args.only_generalisation:
            local_max_nodes = min(args.generalisation_max_nodes, args.max_repairable + 2)
            local_min_nodes = min(args.generalisation_min_nodes, local_max_nodes)

        generate_one(
            out_path=out_path,
            family=family,
            n_instances=n_instances,
            seed=args.seed + idx * 17,
            h=args.H,
            workers=workers,
            mode=mode,
            split_name=split_name,
            min_nodes=local_min_nodes,
            max_nodes=local_max_nodes,
            max_repairable=args.max_repairable if args.only_generalisation else None,
        )

    print("\nAll V7 datasets generated successfully.")


if __name__ == "__main__":
    main()
