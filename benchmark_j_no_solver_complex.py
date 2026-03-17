import argparse
import json
import random
import time
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential, Sigmoid
from torch_geometric.nn import GINEConv, global_mean_pool
from GraphSAGE_J_predictor import GraphSAGE_JStar_Predictor


class GINE_JStar_Predictor(torch.nn.Module):
    def __init__(self, num_node_features=9, hidden_dim=64, edge_dim=1):
        super().__init__()
        nn1 = Sequential(
            Linear(num_node_features, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
        )
        self.conv1 = GINEConv(nn1, edge_dim=edge_dim)

        nn2 = Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
        )
        self.conv2 = GINEConv(nn2, edge_dim=edge_dim)

        self.mlp_readout = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 1),
            Sigmoid(),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x_graph = global_mean_pool(x, batch)
        return self.mlp_readout(x_graph)


PROFILE_PARAMS = {
    "fragile": {"p": (0.23, 0.36), "c": (1.0, 4.0), "aggr": 1.20},
    "stable": {"p": (0.06, 0.14), "c": (5.0, 10.0), "aggr": 0.45},
    "volatile": {"p": (0.14, 0.28), "c": (2.0, 7.0), "aggr": 0.85},
    "bottleneck": {"p": (0.18, 0.32), "c": (3.0, 9.0), "aggr": 1.35},
}


def _sample_profile(rng: random.Random) -> str:
    profiles = ["fragile", "stable", "volatile", "bottleneck"]
    weights = [0.30, 0.20, 0.30, 0.20]
    return rng.choices(profiles, weights=weights, k=1)[0]


def _add_node_with_profile(
    G: nx.DiGraph,
    node_id: int,
    profile: str,
    motif: str,
):
    G.add_node(node_id)
    G.nodes[node_id]["profile"] = profile
    G.nodes[node_id]["motif"] = motif


def _add_parallel_path(G, rng, next_id, max_add, edge):
    if max_add < 2:
        return next_id, 0, "parallel_path"
    u, v = edge
    a, b = next_id, next_id + 1
    _add_node_with_profile(G, a, _sample_profile(rng), "parallel_path")
    _add_node_with_profile(G, b, _sample_profile(rng), "parallel_path")
    G.add_edge(u, a)
    G.add_edge(a, b)
    G.add_edge(b, v)
    return next_id + 2, 2, "parallel_path"


def _add_diamond(G, rng, next_id, max_add, edge):
    if max_add < 3:
        return next_id, 0, "diamond"
    u, v = edge
    a, b, c = next_id, next_id + 1, next_id + 2
    _add_node_with_profile(G, a, _sample_profile(rng), "diamond")
    _add_node_with_profile(G, b, _sample_profile(rng), "diamond")
    _add_node_with_profile(G, c, _sample_profile(rng), "diamond")
    G.add_edge(u, a)
    G.add_edge(u, b)
    G.add_edge(a, c)
    G.add_edge(b, c)
    G.add_edge(c, v)
    return next_id + 3, 3, "diamond"


def _add_detour_cycle(G, rng, next_id, max_add, edge):
    if max_add < 3:
        return next_id, 0, "detour_cycle"
    u, v = edge
    a, b, c = next_id, next_id + 1, next_id + 2
    _add_node_with_profile(G, a, _sample_profile(rng), "detour_cycle")
    _add_node_with_profile(G, b, _sample_profile(rng), "detour_cycle")
    _add_node_with_profile(G, c, _sample_profile(rng), "detour_cycle")
    G.add_edge(u, a)
    G.add_edge(a, b)
    G.add_edge(b, c)
    G.add_edge(c, v)
    G.add_edge(c, a)
    if rng.random() < 0.6:
        G.add_edge(u, c)
    return next_id + 3, 3, "detour_cycle"


def _add_hub_spokes(G, rng, next_id, max_add, edge):
    if max_add < 4:
        return next_id, 0, "hub_spokes"
    u, v = edge
    h, s1, s2, s3 = next_id, next_id + 1, next_id + 2, next_id + 3
    _add_node_with_profile(G, h, "bottleneck", "hub_spokes")
    _add_node_with_profile(G, s1, _sample_profile(rng), "hub_spokes")
    _add_node_with_profile(G, s2, _sample_profile(rng), "hub_spokes")
    _add_node_with_profile(G, s3, _sample_profile(rng), "hub_spokes")
    G.add_edge(u, h)
    G.add_edge(h, s1)
    G.add_edge(h, s2)
    G.add_edge(h, s3)
    G.add_edge(s1, v)
    G.add_edge(s2, v)
    G.add_edge(s3, v)
    return next_id + 4, 4, "hub_spokes"


def _add_ladder(G, rng, next_id, max_add, edge):
    if max_add < 4:
        return next_id, 0, "ladder"
    u, v = edge
    a, b, c, d = next_id, next_id + 1, next_id + 2, next_id + 3
    _add_node_with_profile(G, a, _sample_profile(rng), "ladder")
    _add_node_with_profile(G, b, _sample_profile(rng), "ladder")
    _add_node_with_profile(G, c, _sample_profile(rng), "ladder")
    _add_node_with_profile(G, d, _sample_profile(rng), "ladder")
    G.add_edge(u, a)
    G.add_edge(a, b)
    G.add_edge(b, v)
    G.add_edge(u, c)
    G.add_edge(c, d)
    G.add_edge(d, v)
    G.add_edge(a, c)
    G.add_edge(b, d)
    return next_id + 4, 4, "ladder"


def _compute_pi_heuristic(
    G: nx.DiGraph,
    terminals: List[int],
    repairable_nodes: List[int],
    node_features: Dict[int, Dict[str, float]],
    B: float,
):
    source, target = terminals
    score = {}
    for n in repairable_nodes:
        feat = node_features[n]
        dist = feat["distance_to_target"]
        if dist >= 999:
            dist = 50
        profile = G.nodes[n].get("profile", "volatile")
        aggr = PROFILE_PARAMS[profile]["aggr"]
        near_target = 1.0 / (1.0 + dist)
        path_weight = 1.0 + 0.20 * feat["out_degree"] + 0.10 * feat["in_degree"]
        utility = (feat["p_fail"] * path_weight * (1.0 + near_target)) / max(feat["c_cost"], 0.2)
        score[n] = max(1e-8, utility * aggr)

    max_s = max(score.values()) if score else 1.0
    base = {n: min(0.98, (score[n] / max_s) ** 0.85) for n in repairable_nodes}

    def budget_used(mult):
        return sum(node_features[n]["c_cost"] * min(0.98, mult * base[n]) for n in repairable_nodes)

    lo, hi = 0.0, 8.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if budget_used(mid) > B:
            hi = mid
        else:
            lo = mid

    mult = lo
    pi = {n: min(0.98, mult * base[n]) for n in repairable_nodes}

    # Petite correction gloutonne si budget restant.
    spent = sum(node_features[n]["c_cost"] * pi[n] for n in repairable_nodes)
    remaining = max(0.0, B - spent)
    order = sorted(repairable_nodes, key=lambda n: score[n], reverse=True)
    for n in order:
        if remaining <= 1e-8:
            break
        c = node_features[n]["c_cost"]
        room = 0.98 - pi[n]
        if room <= 1e-8:
            continue
        add = min(room, remaining / c)
        pi[n] += add
        remaining -= add * c

    y = []
    for n in sorted(G.nodes()):
        if n in terminals:
            y.append(0.0)
        else:
            y.append(float(pi[n]))
    return y


def _simulate_monte_carlo(instance: Dict, n_sims: int, rng_seed: int) -> float:
    H = int(instance["H"])
    nodes = instance["graph"]["nodes"]
    edges = instance["graph"]["edges"]
    source, target = instance["terminals"]

    rep_nodes = instance.get("repairable_nodes", [])
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    p_fail = np.array([instance["x"][node_to_idx[n]][0] for n in rep_nodes], dtype=float)
    pi = np.array([instance["y"][node_to_idx[n]] for n in rep_nodes], dtype=float)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    np.random.seed(rng_seed)
    states = np.ones((n_sims, len(rep_nodes)), dtype=bool)
    downtime_counts = np.zeros(n_sims, dtype=float)

    for _ in range(H):
        rands = np.random.rand(n_sims, len(rep_nodes))
        to_down = states & (rands < p_fail)
        to_up = (~states) & (rands < pi)
        states = (states & ~to_down) | to_up

        for i in range(n_sims):
            up_rep = [rep_nodes[j] for j, is_up in enumerate(states[i]) if is_up]
            valid_nodes = set(up_rep) | {source, target}
            sub_g = G.subgraph(valid_nodes)
            connected = nx.has_path(sub_g, source, target) if sub_g.number_of_nodes() > 1 else False
            if not connected:
                downtime_counts[i] += 1.0

    return float(np.mean(downtime_counts) / H)


def _normalize_x_for_model(x_tensor: torch.Tensor) -> torch.Tensor:
    x = x_tensor.clone()
    x[:, 1] = x[:, 1] / 10.0
    x[:, 4] = x[:, 4] / 15.0
    x[:, 5] = x[:, 5] / 15.0
    x[:, 6] = x[:, 6] / 15.0
    x[:, 7] = x[:, 7] / 65.0
    x[:, 8] = x[:, 8] / 25.0
    return x


def _predict_j(model: torch.nn.Module, instance: Dict, device: torch.device) -> Tuple[float, float]:
    nodes = instance["graph"]["nodes"]
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = instance["graph"]["edges"]

    x = torch.tensor(instance["x"], dtype=torch.float32, device=device)
    x = _normalize_x_for_model(x)

    mapped_edges = [[node_to_idx[u], node_to_idx[v]] for u, v in edges if u in node_to_idx and v in node_to_idx]
    if mapped_edges:
        edge_index = torch.tensor(mapped_edges, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float32, device=device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        pred = model(x, edge_index, edge_attr, batch)
        dt = (time.perf_counter() - t0) * 1000.0
    return float(pred.view(-1).item()), float(dt)


def generate_complex_instance(min_nodes: int, max_nodes: int, horizon: int, seed: int) -> Dict:
    rng = random.Random(seed)
    n_target = rng.randint(min_nodes, max_nodes)

    G = nx.DiGraph()
    source, target = 0, 1
    G.add_node(source, profile="terminal", motif="terminal")
    G.add_node(target, profile="terminal", motif="terminal")

    # Backbone principal pour garantir une route S->T de base.
    next_id = 2
    backbone = [source]
    backbone_len = max(8, min(16, n_target // 3))
    for _ in range(backbone_len):
        if next_id >= n_target:
            break
        _add_node_with_profile(G, next_id, _sample_profile(rng), "backbone")
        backbone.append(next_id)
        next_id += 1
    backbone.append(target)
    for u, v in zip(backbone[:-1], backbone[1:]):
        G.add_edge(u, v)
        if rng.random() < 0.18 and u != source:
            G.add_edge(v, u)

    motif_builders = [
        _add_parallel_path,
        _add_diamond,
        _add_detour_cycle,
        _add_hub_spokes,
        _add_ladder,
    ]
    motif_weights = [0.24, 0.22, 0.20, 0.17, 0.17]
    motifs_used = []

    # Ajout de motifs sur des arêtes existantes pour hétérogénéité structurelle.
    while next_id < n_target:
        available_edges = [e for e in G.edges() if e[0] != target and e[1] != source]
        if not available_edges:
            break
        anchor = rng.choice(available_edges)
        max_add = n_target - next_id
        builder = rng.choices(motif_builders, weights=motif_weights, k=1)[0]
        new_next, added, motif_name = builder(G, rng, next_id, max_add, anchor)
        if added <= 0:
            _add_node_with_profile(G, next_id, _sample_profile(rng), "fallback_branch")
            G.add_edge(anchor[0], next_id)
            G.add_edge(next_id, anchor[1])
            next_id += 1
            motifs_used.append("fallback_branch")
        else:
            next_id = new_next
            motifs_used.append(motif_name)

    # Arêtes de couplage inter-branches pour des comportements de panne plus variés.
    all_nodes = [n for n in G.nodes() if n not in (source, target)]
    extra_links = max(3, len(all_nodes) // 8)
    for _ in range(extra_links):
        u, v = rng.sample(all_nodes, 2)
        if u != v and not G.has_edge(u, v):
            if rng.random() < 0.55:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)

    terminals = [source, target]
    repairable_nodes = [n for n in G.nodes() if n not in terminals]

    node_features = {}
    for n in sorted(G.nodes()):
        if n in terminals:
            p_fail = 0.0
            c_cost = 0.0
        else:
            profile = G.nodes[n].get("profile", "volatile")
            p_rng = PROFILE_PARAMS[profile]["p"]
            c_rng = PROFILE_PARAMS[profile]["c"]
            p_fail = round(rng.uniform(*p_rng), 3)
            c_cost = round(rng.uniform(*c_rng), 2)

        try:
            dist = nx.shortest_path_length(G, source=n, target=target)
        except nx.NetworkXNoPath:
            dist = 999

        node_features[n] = {
            "p_fail": float(p_fail),
            "c_cost": float(c_cost),
            "is_source": float(n == source),
            "is_target": float(n == target),
            "in_degree": float(G.in_degree(n)),
            "out_degree": float(G.out_degree(n)),
            "distance_to_target": float(dist),
        }

    C_total = float(sum(node_features[n]["c_cost"] for n in repairable_nodes))
    mean_risk = np.mean([node_features[n]["p_fail"] for n in repairable_nodes]) if repairable_nodes else 0.2
    alpha = float(np.clip(rng.gauss(0.45 + 0.60 * (mean_risk - 0.18), 0.12), 0.16, 0.88))
    B = round(alpha * max(C_total, 1.0), 2)

    y = _compute_pi_heuristic(G, terminals, repairable_nodes, node_features, B)

    x = []
    for n in sorted(G.nodes()):
        feat = node_features[n]
        x.append(
            [
                feat["p_fail"],
                feat["c_cost"],
                feat["is_source"],
                feat["is_target"],
                feat["in_degree"],
                feat["out_degree"],
                feat["distance_to_target"],
                float(B),
                float(horizon),
            ]
        )

    return {
        "topology_type": "complex_motif_mixed",
        "motifs_used": motifs_used,
        "graph": {
            "nodes": sorted(G.nodes()),
            "edges": list(G.edges()),
            "is_directed": True,
        },
        "terminals": terminals,
        "repairable_nodes": repairable_nodes,
        "x": x,
        "y": y,
        "B": float(B),
        "C_total": float(C_total),
        "H": int(horizon),
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
    }


def _load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = GINE_JStar_Predictor().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # Compatibilite DataParallel eventuelle.
        remapped = {}
        for k, v in state.items():
            remapped[k[7:]] = v if k.startswith("module.") else v
        model.load_state_dict(remapped)
    return model


def _load_graphsage_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = GraphSAGE_JStar_Predictor().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state)
    except RuntimeError:
        remapped = {}
        for k, v in state.items():
            remapped[k[7:] if k.startswith("module.") else k] = v
        model.load_state_dict(remapped)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark J sans solveur: generation complexe + comparaison GNN vs Monte-Carlo"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint du modele J")
    parser.add_argument("--model-type", type=str, default="gine", choices=["gine", "graphsage"],
                        help="Architecture du modele : gine (défaut) ou graphsage")
    parser.add_argument("--n-instances", type=int, default=40, help="Nombre d'instances")
    parser.add_argument("--min-nodes", type=int, default=30, help="Taille min des graphes")
    parser.add_argument("--max-nodes", type=int, default=55, help="Taille max des graphes")
    parser.add_argument("--horizon", type=int, default=25, help="Horizon temporel H")
    parser.add_argument("--mc-sims", type=int, default=3000, help="Nombre de simulations Monte-Carlo par graphe")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--out-json", type=str, default=None, help="Fichier de sortie JSON (défaut : benchmark_j_complex_<model>.json)")
    parser.add_argument("--save-instances", action="store_true", help="Sauvegarde complete des instances dans le JSON")
    args = parser.parse_args()
    if args.out_json is None:
        args.out_json = f"benchmark_j_complex_{args.model_type}.json"

    if args.min_nodes < 30:
        raise ValueError("--min-nodes doit etre >= 30 pour respecter le cahier des charges")
    if args.max_nodes < args.min_nodes:
        raise ValueError("--max-nodes doit etre >= --min-nodes")

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
    sq_errors = []
    pred_times = []
    mc_times = []
    node_counts = []
    edge_counts = []

    print(
        f"Benchmark sans solveur | instances={args.n_instances} | "
        f"nodes=[{args.min_nodes},{args.max_nodes}] | H={args.horizon} | mc_sims={args.mc_sims}"
    )

    for i in range(args.n_instances):
        inst_seed = args.seed + i * 7919
        inst = generate_complex_instance(args.min_nodes, args.max_nodes, args.horizon, inst_seed)

        pred_j, pred_ms = _predict_j(model, inst, device)

        t0_mc = time.perf_counter()
        mc_j = _simulate_monte_carlo(inst, n_sims=args.mc_sims, rng_seed=inst_seed + 123)
        mc_ms = (time.perf_counter() - t0_mc) * 1000.0

        err = abs(pred_j - mc_j)
        rows.append(
            {
                "id": i,
                "seed": inst_seed,
                "n_nodes": inst["n_nodes"],
                "n_edges": inst["n_edges"],
                "motifs_used": inst["motifs_used"],
                "pred_j": float(pred_j),
                "mc_j": float(mc_j),
                "abs_error": float(err),
                "pred_time_ms": float(pred_ms),
                "mc_time_ms": float(mc_ms),
                "speedup_mc_over_model": float(mc_ms / max(pred_ms, 1e-9)),
                "instance": inst if args.save_instances else None,
            }
        )
        abs_errors.append(err)
        sq_errors.append(err * err)
        pred_times.append(pred_ms)
        mc_times.append(mc_ms)
        node_counts.append(inst["n_nodes"])
        edge_counts.append(inst["n_edges"])

        if (i + 1) % max(1, args.n_instances // 10) == 0:
            print(
                f"[{i+1}/{args.n_instances}] "
                f"MAE courant={np.mean(abs_errors):.4f} | "
                f"pred_ms={np.mean(pred_times):.2f} | mc_ms={np.mean(mc_times):.2f}"
            )

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    corr = float(np.corrcoef([r["pred_j"] for r in rows], [r["mc_j"] for r in rows])[0, 1]) if len(rows) > 1 else 0.0
    speedup = float(np.mean([r["speedup_mc_over_model"] for r in rows]))

    summary = {
        "config": {
            "checkpoint": args.checkpoint,
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
            "mean_pred_time_ms": float(np.mean(pred_times)),
            "mean_mc_time_ms": float(np.mean(mc_times)),
            "mean_speedup_mc_over_model": speedup,
            "mean_nodes": float(np.mean(node_counts)),
            "mean_edges": float(np.mean(edge_counts)),
        },
        "rows": rows,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Benchmark termine (sans solveur)")
    print(f"MAE(pred, MC): {mae:.5f} | RMSE: {rmse:.5f} | Corr: {corr:.4f}")
    print(f"Temps moyen modele: {np.mean(pred_times):.3f} ms")
    print(f"Temps moyen Monte-Carlo: {np.mean(mc_times):.3f} ms")
    print(f"Accelaration moyenne (MC / modele): x{speedup:.1f}")
    print(f"Resultats ecrits dans: {args.out_json}")


if __name__ == "__main__":
    main()
