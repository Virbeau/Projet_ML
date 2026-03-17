"""
Benchmark GINE_B_repartition sans solveur.

Génère des graphes complexes (>=30 noeuds, motifs hétérogènes), applique
trois politiques naïves de répartition du budget et la prédiction GINE_B,
simule chaque politique avec Monte-Carlo, puis compare les J obtenus.

Méthodes comparées
------------------
1. naive_uniform        : même pi pour tous les noeuds réparables
2. naive_risk_prop      : pi proportionnel au risque p_fail de chaque noeud
3. naive_greedy_utility : tri par score p_fail/c_cost, allocation gloutonne
4. gine_b               : prédiction du réseau GINE_B
"""

import argparse
import json
import random
import time
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from torch.nn import Linear, ReLU, Sequential, Sigmoid
from torch_geometric.nn import GINEConv, global_add_pool
from GraphSAGE_B_repartition import GraphSAGE_Allocation_Predictor

from benchmark_j_no_solver_complex import generate_complex_instance, PROFILE_PARAMS


# ---------------------------------------------------------------------------
# Architecture GINE_B (identique à GINE_B_repartition.py)
# ---------------------------------------------------------------------------
class GINE_Allocation_Predictor(torch.nn.Module):
    def __init__(self, num_node_features=9, hidden_dim=64, edge_dim=1):
        super().__init__()
        self.conv1 = GINEConv(
            nn=Sequential(
                Linear(num_node_features, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            ),
            edge_dim=edge_dim,
        )
        self.conv2 = GINEConv(
            nn=Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            ),
            edge_dim=edge_dim,
        )
        self.mlp_readout = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 1),
            Sigmoid(),
        )

    def forward(self, x, edge_index, edge_attr, batch, B_total, terminal_mask=None, c_cost=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        pi_raw = self.mlp_readout(x).squeeze(-1)

        if terminal_mask is not None:
            pi_raw = pi_raw * (~terminal_mask.bool()).float()

        if c_cost is None:
            c_cost = torch.ones_like(pi_raw)

        node_expenses = pi_raw * c_cost
        total_expenses = global_add_pool(node_expenses.view(-1, 1), batch).view(-1)

        B_b = B_total[batch]
        exp_b = total_expenses[batch]
        ratio = torch.clamp(B_b / (exp_b + 1e-12), max=1.0)
        return pi_raw * ratio


# ---------------------------------------------------------------------------
# Chargement du checkpoint GINE_B
# ---------------------------------------------------------------------------
def _load_gine_b(checkpoint_path: str, device: torch.device) -> GINE_Allocation_Predictor:
    model = GINE_Allocation_Predictor().to(device)
    raw = torch.load(checkpoint_path, map_location=device)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    # Compatibilité DataParallel éventuelle
    remapped = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(remapped)
    model.eval()
    return model


def _load_graphsage_b(checkpoint_path: str, device: torch.device) -> GraphSAGE_Allocation_Predictor:
    model = GraphSAGE_Allocation_Predictor().to(device)
    raw = torch.load(checkpoint_path, map_location=device)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    remapped = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(remapped)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Préparation des inputs pour GINE_B (même normalisation que l'entraînement)
# ---------------------------------------------------------------------------
def _prep_gine_b_inputs(inst: Dict, device: torch.device):
    nodes = inst["graph"]["nodes"]
    edges = inst["graph"]["edges"]
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    x = torch.tensor(inst["x"], dtype=torch.float32)
    x[:, 1] = x[:, 1] / 10.0
    x[:, 4] = x[:, 4] / 15.0
    x[:, 5] = x[:, 5] / 15.0
    x[:, 6] = x[:, 6] / 15.0
    x[:, 7] = x[:, 7] / 65.0
    x[:, 8] = x[:, 8] / 25.0

    mapped = [
        [node_to_idx[u], node_to_idx[v]]
        for u, v in edges
        if u in node_to_idx and v in node_to_idx
    ]
    if mapped:
        edge_index = torch.tensor(mapped, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float32)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    B_total = torch.tensor([inst["B"]], dtype=torch.float32)

    terminal_mask = torch.zeros(x.size(0), dtype=torch.bool)
    for t in inst["terminals"]:
        if t in node_to_idx:
            terminal_mask[node_to_idx[t]] = True

    # c_cost brut (col 1 de x avant normalisation)
    c_cost = torch.tensor([row[1] for row in inst["x"]], dtype=torch.float32)

    return (
        x.to(device),
        edge_index.to(device),
        edge_attr.to(device),
        batch.to(device),
        B_total.to(device),
        terminal_mask.to(device),
        c_cost.to(device),
        node_to_idx,
    )


# ---------------------------------------------------------------------------
# Inférence GINE_B → dict {node_id: pi_value}
# ---------------------------------------------------------------------------
def _predict_gine_b(
    model: GINE_Allocation_Predictor,
    inst: Dict,
    device: torch.device,
) -> Tuple[Dict[int, float], float]:
    x, edge_index, edge_attr, batch, B_total, terminal_mask, c_cost, node_to_idx = _prep_gine_b_inputs(inst, device)
    with torch.no_grad():
        t0 = time.perf_counter()
        alloc = model(x, edge_index, edge_attr, batch, B_total, terminal_mask, c_cost)
        inference_ms = (time.perf_counter() - t0) * 1000.0
    alloc_np = alloc.cpu().numpy()
    return {n: float(alloc_np[idx]) for n, idx in node_to_idx.items()}, float(inference_ms)


def _predict_graphsage_b(
    model: GraphSAGE_Allocation_Predictor,
    inst: Dict,
    device: torch.device,
) -> Tuple[Dict[int, float], float]:
    x, edge_index, edge_attr, batch, B_total, terminal_mask, c_cost, node_to_idx = _prep_gine_b_inputs(inst, device)
    with torch.no_grad():
        t0 = time.perf_counter()
        alloc = model(x, edge_index, edge_attr, batch, B_total, terminal_mask, c_cost)
        inference_ms = (time.perf_counter() - t0) * 1000.0
    alloc_np = alloc.cpu().numpy()
    return {n: float(alloc_np[idx]) for n, idx in node_to_idx.items()}, float(inference_ms)


# ---------------------------------------------------------------------------
# Politiques naïves de répartition du budget
# ---------------------------------------------------------------------------
def _project_budget(pi_raw: Dict[int, float], c_cost: Dict[int, float], B: float) -> Dict[int, float]:
    """Projette pi_raw pour respecter sum(pi_i * c_i) <= B."""
    total = sum(pi_raw[n] * c_cost[n] for n in pi_raw)
    if total <= B + 1e-10:
        return {n: min(0.98, v) for n, v in pi_raw.items()}
    # Recherche binaire sur le multiplicateur
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        spent = sum(min(0.98, mid * pi_raw[n]) * c_cost[n] for n in pi_raw)
        if spent > B:
            hi = mid
        else:
            lo = mid
    return {n: min(0.98, lo * pi_raw[n]) for n in pi_raw}


def naive_uniform(inst: Dict) -> Dict[int, float]:
    """
    Naïf 1 — Uniforme : même probabilité pi pour tous les noeuds réparables.
    La valeur commune est B / sum(c_i), plafonnée à 0.98.
    """
    rep = inst["repairable_nodes"]
    nodes = inst["graph"]["nodes"]
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    B = inst["B"]
    c_cost = {n: inst["x"][node_to_idx[n]][1] for n in rep}
    total_cost = sum(c_cost.values()) or 1.0
    pi = min(0.98, B / total_cost)
    return {n: pi for n in rep}


def naive_risk_prop(inst: Dict) -> Dict[int, float]:
    """
    Naïf 2 — Proportionnel au risque : pi_i proportionnel à p_fail_i.
    Projette pour respecter le budget.
    Plus de budget va aux noeuds les plus risqués.
    """
    rep = inst["repairable_nodes"]
    nodes = inst["graph"]["nodes"]
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    B = inst["B"]
    c_cost = {n: inst["x"][node_to_idx[n]][1] for n in rep}
    p_fail = {n: inst["x"][node_to_idx[n]][0] for n in rep}

    max_p = max(p_fail.values(), default=1.0) or 1.0
    pi_raw = {n: p_fail[n] / max_p for n in rep}  # normalise dans [0,1]
    return _project_budget(pi_raw, c_cost, B)


def naive_greedy_utility(inst: Dict) -> Dict[int, float]:
    """
    Naïf 3 — Utilité gloutonne : score = p_fail / c_cost.
    Trie les noeuds par score décroissant, alloue pi=0.98 au maximum jusqu'à
    épuisement du budget. Allocation partielle pour le dernier noeud.
    """
    rep = inst["repairable_nodes"]
    nodes = inst["graph"]["nodes"]
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    B = inst["B"]
    c_cost = {n: inst["x"][node_to_idx[n]][1] for n in rep}
    p_fail = {n: inst["x"][node_to_idx[n]][0] for n in rep}

    score = {n: p_fail[n] / max(c_cost[n], 0.1) for n in rep}
    order = sorted(rep, key=lambda n: score[n], reverse=True)

    pi = {n: 0.0 for n in rep}
    remaining = B
    for n in order:
        if remaining <= 1e-8:
            break
        c = c_cost[n]
        max_pi = min(0.98, remaining / c)
        pi[n] = max_pi
        remaining -= max_pi * c
    return pi


# ---------------------------------------------------------------------------
# Simulation Monte-Carlo avec une politique donnée
# ---------------------------------------------------------------------------
def _simulate_mc_with_policy(
    inst: Dict,
    pi_dict: Dict[int, float],
    n_sims: int,
    rng_seed: int,
) -> float:
    """
    Simule n_sims trajectoires avec la politique pi_dict.
    Retourne J = downtime moyen / H ∈ [0, 1].
    """
    H = int(inst["H"])
    nodes = inst["graph"]["nodes"]
    edges = inst["graph"]["edges"]
    source, target = inst["terminals"]
    rep_nodes = inst["repairable_nodes"]
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    p_fail_arr = np.array([inst["x"][node_to_idx[n]][0] for n in rep_nodes], dtype=float)
    pi_arr = np.array([pi_dict.get(n, 0.0) for n in rep_nodes], dtype=float)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    np.random.seed(rng_seed)
    states = np.ones((n_sims, len(rep_nodes)), dtype=bool)
    downtime = np.zeros(n_sims, dtype=float)

    for _ in range(H):
        rands = np.random.rand(n_sims, len(rep_nodes))
        to_down = states & (rands < p_fail_arr)
        to_up = (~states) & (rands < pi_arr)
        states = (states & ~to_down) | to_up

        for i in range(n_sims):
            up_rep = [rep_nodes[j] for j, up in enumerate(states[i]) if up]
            valid = set(up_rep) | {source, target}
            sub = G.subgraph(valid)
            connected = sub.number_of_nodes() > 1 and nx.has_path(sub, source, target)
            if not connected:
                downtime[i] += 1.0

    return float(np.mean(downtime) / H)


def _budget_spent(pi_dict: Dict[int, float], inst: Dict) -> float:
    nodes = inst["graph"]["nodes"]
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    rep = inst["repairable_nodes"]
    return sum(pi_dict.get(n, 0.0) * inst["x"][node_to_idx[n]][1] for n in rep)


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark GINE_B sans solveur: graphes complexes >=30 noeuds, "
            "3 politiques naïves vs GINE_B, comparaison par Monte-Carlo."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint GINE_B (.pt)")
    parser.add_argument("--model-type", type=str, default="gine", choices=["gine", "graphsage"],
                        help="Architecture du modele : gine (défaut) ou graphsage")
    parser.add_argument("--n-instances", type=int, default=40,
                        help="Nombre d'instances (défaut 40)")
    parser.add_argument("--min-nodes", type=int, default=30,
                        help="Taille min des graphes (défaut 30)")
    parser.add_argument("--max-nodes", type=int, default=55,
                        help="Taille max des graphes (défaut 55)")
    parser.add_argument("--horizon", type=int, default=25,
                        help="Horizon H (défaut 25)")
    parser.add_argument("--mc-sims", type=int, default=3000,
                        help="Simulations MC par instance (défaut 3000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=str, default=None,
                        help="Fichier de sortie JSON (défaut : benchmark_b_complex_<model>.json)")
    parser.add_argument("--save-instances", action="store_true",
                        help="Inclure instances complètes dans le JSON")
    args = parser.parse_args()
    if args.out_json is None:
        args.out_json = f"benchmark_b_complex_{args.model_type}.json"

    if args.min_nodes < 30:
        raise ValueError("--min-nodes doit être >= 30")
    if args.max_nodes < args.min_nodes:
        raise ValueError("--max-nodes doit être >= --min-nodes")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    if args.model_type == "graphsage":
        model = _load_graphsage_b(args.checkpoint, device)
        model_label = "GraphSAGE_B"
    else:
        model = _load_gine_b(args.checkpoint, device)
        model_label = "GINE_B"

    METHODS = ["uniform", "risk_prop", "greedy", "gine_b"]
    J_per_method: Dict[str, List[float]] = {m: [] for m in METHODS}
    spend_per_method: Dict[str, List[float]] = {m: [] for m in METHODS}
    gine_times_ms: List[float] = []
    mc_times_ms: List[float] = []
    rows: List[Dict] = []

    print(
        f"\nBenchmark {model_label} sans solveur | n={args.n_instances} | "
        f"nodes=[{args.min_nodes},{args.max_nodes}] | H={args.horizon} | mc_sims={args.mc_sims}"
    )
    print(f"Checkpoint: {args.checkpoint} | Device: {device}\n")

    for i in range(args.n_instances):
        inst_seed = args.seed + i * 7919
        inst = generate_complex_instance(
            args.min_nodes, args.max_nodes, args.horizon, inst_seed
        )
        B = inst["B"]
        mc_seed = inst_seed + 9999

        # -- Politiques naïves
        pi_uniform = naive_uniform(inst)
        pi_risk = naive_risk_prop(inst)
        pi_greedy = naive_greedy_utility(inst)

        # -- Modèle B
        if args.model_type == "graphsage":
            pi_gine, gine_ms = _predict_graphsage_b(model, inst, device)
        else:
            pi_gine, gine_ms = _predict_gine_b(model, inst, device)
        gine_times_ms.append(gine_ms)

        # -- Simulations Monte-Carlo
        t0_mc = time.perf_counter()
        J_uniform  = _simulate_mc_with_policy(inst, pi_uniform,  args.mc_sims, mc_seed)
        J_risk     = _simulate_mc_with_policy(inst, pi_risk,     args.mc_sims, mc_seed + 1)
        J_greedy   = _simulate_mc_with_policy(inst, pi_greedy,   args.mc_sims, mc_seed + 2)
        J_gine     = _simulate_mc_with_policy(inst, pi_gine,     args.mc_sims, mc_seed + 3)
        mc_ms = (time.perf_counter() - t0_mc) * 1000.0
        mc_times_ms.append(mc_ms / 4.0)  # par simulation

        J_per_method["uniform"].append(J_uniform)
        J_per_method["risk_prop"].append(J_risk)
        J_per_method["greedy"].append(J_greedy)
        J_per_method["gine_b"].append(J_gine)

        for m, pi in zip(METHODS, [pi_uniform, pi_risk, pi_greedy, pi_gine]):
            spend_per_method[m].append(_budget_spent(pi, inst))

        best_naive = min(J_uniform, J_risk, J_greedy)
        gine_wins = J_gine < best_naive
        gine_delta_vs_best_naive = best_naive - J_gine  # positif si GINE_B gagne

        row = {
            "id": i,
            "seed": inst_seed,
            "n_nodes": inst["n_nodes"],
            "n_edges": inst["n_edges"],
            "B": float(B),
            "motifs_used": inst["motifs_used"],
            "J_uniform":   float(J_uniform),
            "J_risk_prop": float(J_risk),
            "J_greedy":    float(J_greedy),
            "J_gine_b":    float(J_gine),
            "J_model_b":   float(J_gine),
            "spend_uniform":   float(spend_per_method["uniform"][-1]),
            "spend_risk_prop": float(spend_per_method["risk_prop"][-1]),
            "spend_greedy":    float(spend_per_method["greedy"][-1]),
            "spend_gine_b":    float(spend_per_method["gine_b"][-1]),
            "spend_model_b":   float(spend_per_method["gine_b"][-1]),
            "gine_wins_vs_best_naive": bool(gine_wins),
            "gine_delta_vs_best_naive": float(gine_delta_vs_best_naive),
            "gine_inference_ms": float(gine_ms),
            "instance": inst if args.save_instances else None,
        }
        rows.append(row)

        if (i + 1) % max(1, args.n_instances // 10) == 0:
            gine_win_so_far = sum(r["gine_wins_vs_best_naive"] for r in rows) / len(rows) * 100
            print(
                f"[{i+1:3d}/{args.n_instances}] "
                f"J_uniform={np.mean(J_per_method['uniform']):.4f} | "
                f"J_risk={np.mean(J_per_method['risk_prop']):.4f} | "
                f"J_greedy={np.mean(J_per_method['greedy']):.4f} | "
                f"J_model={np.mean(J_per_method['gine_b']):.4f} | "
                f"{model_label} wins={gine_win_so_far:.1f}%"
            )

    # -------------------------------------------------------------------------
    # Récapitulatif global
    # -------------------------------------------------------------------------
    def _stats(vals):
        a = np.array(vals)
        return {"mean": float(np.mean(a)), "median": float(np.median(a)),
                "std": float(np.std(a)), "min": float(np.min(a)), "max": float(np.max(a))}

    win_vs_uniform  = sum(r["J_gine_b"] < r["J_uniform"]   for r in rows) / len(rows) * 100
    win_vs_risk     = sum(r["J_gine_b"] < r["J_risk_prop"] for r in rows) / len(rows) * 100
    win_vs_greedy   = sum(r["J_gine_b"] < r["J_greedy"]    for r in rows) / len(rows) * 100
    win_vs_all      = sum(r["gine_wins_vs_best_naive"]      for r in rows) / len(rows) * 100

    # Gain moyen de GINE_B vs chaque méthode (>0 = GINE_B meilleur)
    gain_vs_uniform = float(np.mean([r["J_uniform"]   - r["J_gine_b"] for r in rows]))
    gain_vs_risk    = float(np.mean([r["J_risk_prop"] - r["J_gine_b"] for r in rows]))
    gain_vs_greedy  = float(np.mean([r["J_greedy"]    - r["J_gine_b"] for r in rows]))

    summary = {
        "config": {
            "checkpoint": args.checkpoint,
            "model_type": args.model_type,
            "n_instances":  args.n_instances,
            "min_nodes":    args.min_nodes,
            "max_nodes":    args.max_nodes,
            "horizon":      args.horizon,
            "mc_sims":      args.mc_sims,
            "seed":         args.seed,
            "device":       str(device),
        },
        "J_stats": {
            "uniform":   _stats(J_per_method["uniform"]),
            "risk_prop": _stats(J_per_method["risk_prop"]),
            "greedy":    _stats(J_per_method["greedy"]),
            "gine_b":    _stats(J_per_method["gine_b"]),
        },
        "gine_b_win_rate": {
            "vs_uniform_pct":  float(win_vs_uniform),
            "vs_risk_prop_pct": float(win_vs_risk),
            "vs_greedy_pct":   float(win_vs_greedy),
            "vs_all_naive_pct": float(win_vs_all),
        },
        "gine_b_mean_gain": {
            "vs_uniform":  gain_vs_uniform,
            "vs_risk_prop": gain_vs_risk,
            "vs_greedy":   gain_vs_greedy,
        },
        "timing": {
            "mean_gine_b_inference_ms": float(np.mean(gine_times_ms)),
            "mean_mc_per_policy_ms":    float(np.mean(mc_times_ms)),
        },
        "rows": rows,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # -------------------------------------------------------------------------
    # Affichage tableau comparatif final
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"RÉSULTATS BENCHMARK {model_label} vs POLITIQUES NAÏVES")
    print("=" * 72)
    print(
        f"{'Méthode':<22} {'Mean J':>10} {'Median J':>10} "
        f"{'Std J':>8} {'Min J':>8} {'Max J':>8}"
    )
    print("-" * 72)
    for m, label in [
        ("uniform",   "Naïf 1 — Uniforme"),
        ("risk_prop", "Naïf 2 — Risk-prop"),
        ("greedy",    "Naïf 3 — Greedy"),
        ("gine_b",    model_label),
    ]:
        s = summary["J_stats"][m]
        marker = " ◀" if m == "gine_b" else ""
        print(
            f"{label + marker:<22} {s['mean']:>10.5f} {s['median']:>10.5f} "
            f"{s['std']:>8.5f} {s['min']:>8.5f} {s['max']:>8.5f}"
        )
        print("=" * 72)
        print(f"  {model_label} bat Naïf Uniforme   : {win_vs_uniform:.1f}% des instances"
            f"  (gain moyen J: {gain_vs_uniform:+.5f})")
        print(f"  {model_label} bat Naïf Risk-prop  : {win_vs_risk:.1f}% des instances"
            f"  (gain moyen J: {gain_vs_risk:+.5f})")
        print(f"  {model_label} bat Naïf Greedy     : {win_vs_greedy:.1f}% des instances"
            f"  (gain moyen J: {gain_vs_greedy:+.5f})")
        print(f"  {model_label} bat TOUTES les naïves: {win_vs_all:.1f}% des instances")
    print("-" * 72)
    print(
          f"  Temps inference {model_label}: {np.mean(gine_times_ms):.3f} ms | "
        f"Temps MC (1 pol.): {np.mean(mc_times_ms):.1f} ms"
    )
    print(f"\nRésultats complets: {args.out_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()
