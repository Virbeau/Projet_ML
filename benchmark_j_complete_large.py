import argparse
import json
import random
import time

import networkx as nx
import numpy as np
import torch

from benchmark_j_no_solver_complex import (
    _compute_pi_heuristic,
    _load_model,
    _predict_j,
    _simulate_monte_carlo,
)
from GraphSAGE_J_predictor import GraphSAGE_JStar_Predictor


COMPLETE_PROFILE_PARAMS = {
    "fragile": {"p": (0.04, 0.10), "c": (1.0, 2.8)},
    "stable": {"p": (0.015, 0.05), "c": (3.0, 6.8)},
    "volatile": {"p": (0.035, 0.085), "c": (1.6, 4.4)},
    "bottleneck": {"p": (0.05, 0.12), "c": (2.8, 6.5)},
}


def _load_graphsage_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = GraphSAGE_JStar_Predictor().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state)
    except RuntimeError:
        remapped = {}
        for key, value in state.items():
            remapped[key[7:] if key.startswith("module.") else key] = value
        model.load_state_dict(remapped)
    return model


def _sample_profile(rng, centrality_rank, n_rep):
    if centrality_rank < max(2, n_rep // 7):
        return "bottleneck"
    return rng.choices(["fragile", "stable", "volatile"], weights=[0.28, 0.30, 0.42], k=1)[0]


def _decorate_and_pack(graph, family, horizon, seed):
    rng = random.Random(seed)
    terminals = [n for n, d in graph.nodes(data=True) if d.get("terminal")]
    source = [n for n in terminals if graph.nodes[n].get("terminal") == "source"][0]
    target = [n for n in terminals if graph.nodes[n].get("terminal") == "target"][0]
    repairable_nodes = [n for n in graph.nodes() if n not in (source, target)]

    centrality = nx.betweenness_centrality(graph.to_undirected())
    ranked = sorted(repairable_nodes, key=lambda n: (centrality[n], -graph.in_degree(n), graph.out_degree(n)), reverse=True)
    profile_map = {}
    for rank, node in enumerate(ranked):
        profile_map[node] = _sample_profile(rng, rank, len(repairable_nodes))
        graph.nodes[node]["profile"] = profile_map[node]

    node_features = {}
    for node in sorted(graph.nodes()):
        if node in (source, target):
            p_fail = 0.0
            c_cost = 0.0
        else:
            params = COMPLETE_PROFILE_PARAMS[profile_map[node]]
            p_fail = round(rng.uniform(*params["p"]), 3)
            c_cost = round(rng.uniform(*params["c"]), 2)

        try:
            dist = nx.shortest_path_length(graph, source=node, target=target)
        except nx.NetworkXNoPath:
            dist = 999

        node_features[node] = {
            "p_fail": float(p_fail),
            "c_cost": float(c_cost),
            "is_source": float(node == source),
            "is_target": float(node == target),
            "in_degree": float(graph.in_degree(node)),
            "out_degree": float(graph.out_degree(node)),
            "distance_to_target": float(dist),
        }

    c_total = float(sum(node_features[n]["c_cost"] for n in repairable_nodes))
    mean_risk = float(np.mean([node_features[n]["p_fail"] for n in repairable_nodes]))
    density_bonus = graph.number_of_edges() / max(graph.number_of_nodes() * (graph.number_of_nodes() - 1), 1)
    alpha = float(np.clip(rng.gauss(0.10 + 0.35 * mean_risk + 0.10 * density_bonus, 0.015), 0.06, 0.18))
    budget = round(alpha * max(c_total, 1.0), 2)

    y = _compute_pi_heuristic(graph, [source, target], repairable_nodes, node_features, budget)

    x = []
    for node in sorted(graph.nodes()):
        feat = node_features[node]
        x.append([
            feat["p_fail"],
            feat["c_cost"],
            feat["is_source"],
            feat["is_target"],
            feat["in_degree"],
            feat["out_degree"],
            feat["distance_to_target"],
            float(budget),
            float(horizon),
        ])

    return {
        "topology_type": f"complete_large_{family}",
        "family": family,
        "graph": {
            "nodes": sorted(graph.nodes()),
            "edges": list(graph.edges()),
            "is_directed": True,
        },
        "terminals": [source, target],
        "repairable_nodes": repairable_nodes,
        "x": x,
        "y": y,
        "B": float(budget),
        "C_total": float(c_total),
        "H": int(horizon),
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "motifs_used": [family],
    }


def _build_complete_dag(seed, target_nodes, horizon):
    graph = nx.DiGraph()
    for node in range(target_nodes):
        if node == 0:
            graph.add_node(node, terminal="source")
        elif node == target_nodes - 1:
            graph.add_node(node, terminal="target")
        else:
            graph.add_node(node)

    for src in range(target_nodes - 1):
        for dst in range(src + 1, target_nodes):
            graph.add_edge(src, dst)

    return _decorate_and_pack(graph, "complete", horizon, seed)


def _build_layered_complete(seed, target_nodes, horizon):
    rng = random.Random(seed)
    graph = nx.DiGraph()
    source, target = 0, target_nodes - 1
    graph.add_node(source, terminal="source")
    graph.add_node(target, terminal="target")
    for node in range(1, target_nodes - 1):
        graph.add_node(node)

    n_internal = max(1, target_nodes - 2)
    n_layers = max(4, min(6, target_nodes // 8))
    base = n_internal // n_layers
    extra = n_internal % n_layers

    layers = []
    next_node = 1
    for layer_idx in range(n_layers):
        layer_size = base + (1 if layer_idx < extra else 0)
        current_layer = list(range(next_node, next_node + layer_size))
        if current_layer:
            layers.append(current_layer)
        next_node += layer_size

    if not layers:
        graph.add_edge(source, target)
        return _decorate_and_pack(graph, "layered_complete", horizon, seed)

    for node in layers[0]:
        graph.add_edge(source, node)

    for current_layer, next_layer in zip(layers[:-1], layers[1:]):
        for src in current_layer:
            for dst in next_layer:
                graph.add_edge(src, dst)
        if rng.random() < 0.8:
            for src in current_layer:
                for dst in next_layer[1::2]:
                    graph.add_edge(src, dst)

    for node in layers[-1]:
        graph.add_edge(node, target)

    for idx, layer in enumerate(layers[:-2]):
        jump_layer = layers[idx + 2]
        if rng.random() < 0.65:
            for src in layer[::2]:
                for dst in jump_layer:
                    graph.add_edge(src, dst)

    return _decorate_and_pack(graph, "layered_complete", horizon, seed)


def generate_complete_large_instance(min_nodes, max_nodes, horizon, seed):
    rng = random.Random(seed)
    n_target = rng.randint(min_nodes, max_nodes)
    families = ["complete", "layered_complete"]
    family = families[seed % len(families)]
    if family == "complete":
        return _build_complete_dag(seed, n_target, horizon)
    return _build_layered_complete(seed, n_target, horizon)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark J sans solveur sur graphes complets/denses de grande taille"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model-type", type=str, default="gine", choices=["gine", "graphsage"],
        help="Architecture du modele : gine (défaut) ou graphsage"
    )
    parser.add_argument("--n-instances", type=int, default=40)
    parser.add_argument("--min-nodes", type=int, default=28)
    parser.add_argument("--max-nodes", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--mc-sims", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=str, default=None,
                        help="Fichier de sortie JSON (défaut : benchmark_j_complete_<model>.json)")
    parser.add_argument("--save-instances", action="store_true")
    args = parser.parse_args()

    if args.out_json is None:
        args.out_json = f"benchmark_j_complete_{args.model_type}.json"

    if args.min_nodes < 20:
        raise ValueError("--min-nodes doit être >= 20 pour ce benchmark")
    if args.max_nodes < args.min_nodes:
        raise ValueError("--max-nodes doit être >= --min-nodes")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    if args.model_type == "graphsage":
        model = _load_graphsage_model(args.checkpoint, device)
    else:
        model = _load_model(args.checkpoint, device)

    rows = []
    abs_errors = []

    print(
        f"\nBenchmark J complete-large | n={args.n_instances} | "
        f"nodes=[{args.min_nodes},{args.max_nodes}] | H={args.horizon} | mc_sims={args.mc_sims}"
    )
    print(f"Checkpoint: {args.checkpoint} | Model: {args.model_type} | Device: {device}\n")

    for i in range(args.n_instances):
        inst_seed = args.seed + i * 7919
        inst = generate_complete_large_instance(args.min_nodes, args.max_nodes, args.horizon, inst_seed)

        pred_j, pred_ms = _predict_j(model, inst, device)

        t0 = time.perf_counter()
        mc_j = _simulate_monte_carlo(inst, n_sims=args.mc_sims, rng_seed=inst_seed + 123)
        mc_ms = (time.perf_counter() - t0) * 1000.0

        err = abs(pred_j - mc_j)
        abs_errors.append(err)

        row = {
            "id": i,
            "seed": inst_seed,
            "family": inst.get("family", "unknown"),
            "n_nodes": inst["n_nodes"],
            "n_edges": inst["n_edges"],
            "motifs_used": [inst.get("family", "unknown")],
            "pred_j": float(pred_j),
            "mc_j": float(mc_j),
            "abs_error": float(err),
            "pred_time_ms": float(pred_ms),
            "mc_time_ms": float(mc_ms),
            "speedup_mc_over_model": float(mc_ms / max(pred_ms, 1e-9)),
            "instance": inst if args.save_instances else None,
        }
        rows.append(row)

        if (i + 1) % max(1, args.n_instances // 10) == 0:
            print(
                f"[{i+1:3d}/{args.n_instances}] "
                f"family={row['family']:<16} | pred={pred_j:.4f} | mc={mc_j:.4f} | "
                f"MAE courant={np.mean(abs_errors):.4f}"
            )

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean([(r["pred_j"] - r["mc_j"]) ** 2 for r in rows])))
    corr = float(np.corrcoef([r["pred_j"] for r in rows], [r["mc_j"] for r in rows])[0, 1]) if len(rows) > 1 else 0.0
    speedup = float(np.mean([r["speedup_mc_over_model"] for r in rows]))

    summary = {
        "config": {
            "checkpoint": args.checkpoint,
            "model_type": args.model_type,
            "n_instances": args.n_instances,
            "min_nodes": args.min_nodes,
            "max_nodes": args.max_nodes,
            "horizon": args.horizon,
            "mc_sims": args.mc_sims,
            "seed": args.seed,
            "device": str(device),
        },
        "metrics": {
            "mae_pred_vs_mc": mae,
            "rmse_pred_vs_mc": rmse,
            "pearson_pred_mc": corr,
            "mean_pred_time_ms": float(np.mean([r["pred_time_ms"] for r in rows])),
            "mean_mc_time_ms": float(np.mean([r["mc_time_ms"] for r in rows])),
            "mean_speedup_mc_over_model": speedup,
            "mean_nodes": float(np.mean([r["n_nodes"] for r in rows])),
            "mean_edges": float(np.mean([r["n_edges"] for r in rows])),
        },
        "rows": rows,
    }

    with open(args.out_json, "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 72)
    print("RÉSULTATS BENCHMARK J COMPLETE-LARGE")
    print("=" * 72)
    print(f"MAE pred vs MC         : {mae:.4f}")
    print(f"RMSE pred vs MC        : {rmse:.4f}")
    print(f"Corrélation pred/MC    : {corr:.4f}")
    print(f"Speedup moyen MC/model : x{speedup:.1f}")
    print(f"n_nodes moyen          : {summary['metrics']['mean_nodes']:.2f}")
    print(f"n_edges moyen          : {summary['metrics']['mean_edges']:.2f}")
    print(f"Résultats complets     : {args.out_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()