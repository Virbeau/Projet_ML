#!/usr/bin/env python3
import json
import math
from pathlib import Path
from collections import defaultdict
import statistics
import argparse
import csv

import numpy as np
import networkx as nx
from scipy.stats import spearmanr, t as student_t


def percentile(values, q):
    return float(np.percentile(np.array(values, dtype=float), q))


def safe_corr(a, b):
    if len(a) < 2:
        return None
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    if np.std(a_arr) == 0 or np.std(b_arr) == 0:
        return None
    return float(np.corrcoef(a_arr, b_arr)[0, 1])


def safe_spearman(a, b):
    if len(a) < 2:
        return None
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    if np.std(a_arr) == 0 or np.std(b_arr) == 0:
        return None
    corr, _ = spearmanr(a_arr, b_arr)
    if np.isnan(corr):
        return None
    return float(corr)


def onehot_topology(topologies):
    unique = sorted(set(topologies))
    if len(unique) <= 1:
        return np.zeros((len(topologies), 0), dtype=float), unique, None

    baseline = unique[0]
    cols = []
    for topo in unique[1:]:
        cols.append([1.0 if t == topo else 0.0 for t in topologies])
    return np.array(cols, dtype=float).T, unique, baseline


def partial_corr_with_p(x, y, controls):
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    z_arr = np.array(controls, dtype=float)
    if z_arr.ndim == 1:
        z_arr = z_arr.reshape(-1, 1)

    n = len(x_arr)
    if n < 3:
        return None

    k = z_arr.shape[1]
    if n <= (k + 2):
        return None

    X = np.column_stack([np.ones(n), z_arr])
    beta_x = np.linalg.lstsq(X, x_arr, rcond=None)[0]
    beta_y = np.linalg.lstsq(X, y_arr, rcond=None)[0]
    rx = x_arr - X @ beta_x
    ry = y_arr - X @ beta_y

    if np.std(rx) == 0 or np.std(ry) == 0:
        return None

    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - k - 2
    t_val = r * math.sqrt(dof / max(1e-15, (1 - r * r)))
    p_val = float(2 * (1 - student_t.cdf(abs(t_val), dof)))

    return {
        "r": r,
        "p_value": p_val,
        "dof": int(dof),
    }


def standardized_beta(target, main_feature, controls):
    y = np.array(target, dtype=float)
    x_main = np.array(main_feature, dtype=float)
    z = np.array(controls, dtype=float)

    if z.ndim == 1:
        z = z.reshape(-1, 1)

    X = np.column_stack([x_main, z])
    stds = np.std(X, axis=0)
    if np.any(stds == 0) or np.std(y) == 0:
        return None

    Xz = (X - np.mean(X, axis=0)) / stds
    yz = (y - np.mean(y)) / np.std(y)
    Xd = np.column_stack([np.ones(len(yz)), Xz])
    beta = np.linalg.lstsq(Xd, yz, rcond=None)[0]
    return float(beta[1])


def allocation_quality_metrics(inst):
    nodes = inst.get("graph", {}).get("nodes", [])
    edges = inst.get("graph", {}).get("edges", [])
    y = inst.get("y", [])
    repairable = inst.get("repairable_nodes", [])
    is_directed = bool(inst.get("graph", {}).get("is_directed", True))

    if not nodes or not repairable or not y:
        return {
            "allocation_spearman_centrality": 0.0,
            "allocation_efficiency_score": 0.0,
            "budget_concentration": 0.0,
            "budget_active": False,
        }

    G = nx.DiGraph() if is_directed else nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    try:
        centrality = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        centrality = {n: 0.0 for n in nodes}

    y_map = {nodes[i]: float(y[i]) for i in range(min(len(nodes), len(y)))}
    budgets = []
    centralities = []

    for node in repairable:
        if node in y_map and node in centrality:
            budgets.append(y_map[node])
            centralities.append(float(centrality[node]))

    if len(budgets) < 2:
        return {
            "allocation_spearman_centrality": 0.0,
            "allocation_efficiency_score": 0.0,
            "budget_concentration": 0.0,
            "budget_active": False,
        }

    budgets_arr = np.array(budgets, dtype=float)
    cent_arr = np.array(centralities, dtype=float)
    budget_active = bool(np.sum(budgets_arr) > 0)

    if np.std(budgets_arr) == 0 or np.std(cent_arr) == 0:
        return {
            "allocation_spearman_centrality": 0.0,
            "allocation_efficiency_score": 0.0,
            "budget_concentration": 0.0,
            "budget_active": budget_active,
        }

    s_corr = safe_spearman(budgets_arr, cent_arr)
    s_corr = float(s_corr) if s_corr is not None else 0.0

    if not budget_active:
        return {
            "allocation_spearman_centrality": s_corr,
            "allocation_efficiency_score": 0.0,
            "budget_concentration": 0.0,
            "budget_active": budget_active,
        }

    b_norm = budgets_arr / (np.sum(budgets_arr) + 1e-12)
    if np.max(cent_arr) > 0:
        c_norm = cent_arr / np.max(cent_arr)
    else:
        c_norm = cent_arr

    efficiency = b_norm * c_norm
    top_n = min(3, len(budgets_arr))
    top_idx = np.argsort(budgets_arr)[-top_n:][::-1]
    top_eff = float(np.mean(efficiency[top_idx]))

    return {
        "allocation_spearman_centrality": s_corr,
        "allocation_efficiency_score": top_eff,
        "budget_concentration": float(np.std(b_norm)),
        "budget_active": budget_active,
    }


def family_from_topology(topology_type):
    if topology_type.startswith("mesh"):
        return "mesh"
    if topology_type.startswith("sp"):
        return "sp"
    if topology_type.startswith("er"):
        return "er"
    return "other"


def summarize_j(values):
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def rate_below(values, threshold):
    arr = np.array(values, dtype=float)
    return float(np.mean(arr <= threshold))


def rate_above(values, threshold):
    arr = np.array(values, dtype=float)
    return float(np.mean(arr >= threshold))


def main():
    parser = argparse.ArgumentParser(description="Analyse statistique détaillée d'un dataset hybride")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_hybrid_mesh_sp_er_v2_1000.json",
        help="Chemin vers le dataset JSON à analyser"
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Préfixe des fichiers de sortie (défaut: dérivé du nom du dataset)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if args.out_prefix:
        prefix = args.out_prefix
    else:
        prefix = dataset_path.stem

    out_json = Path(f"{prefix}_analysis.json")
    out_txt = Path(f"{prefix}_analysis_summary.txt")
    out_influence_json = Path(f"{prefix}_influence_variables.json")
    out_influence_csv = Path(f"{prefix}_influence_variables.csv")

    with dataset_path.open("r") as f:
        data = json.load(f)

    instances = data["instances"]

    B_values = [float(i["B"]) for i in instances]
    H_values = [int(i["H"]) for i in instances]
    N_values = [int(i["n_nodes"]) for i in instances]
    E_values = [int(i["n_edges"]) for i in instances]
    J_values = [float(i["J_star"]) for i in instances]

    # Quantiles de segmentation
    B_q25 = percentile(B_values, 25)
    B_q75 = percentile(B_values, 75)
    H_q25 = percentile(H_values, 25)
    H_q75 = percentile(H_values, 75)
    N_q25 = percentile(N_values, 25)
    N_q75 = percentile(N_values, 75)

    groups_topology = defaultdict(list)
    groups_family = defaultdict(list)
    groups_H = defaultdict(list)
    groups_N = defaultdict(list)
    groups_B_bin = defaultdict(list)
    groups_density_bin = defaultdict(list)
    groups_segment = defaultdict(list)

    full_path_exists = []
    directed_flags = []
    terminal_distances = []
    topology_values = []
    allocation_spearman_values = []
    allocation_efficiency_values = []
    budget_concentration_values = []
    budget_active_flags = []
    influence_rows = []

    for inst in instances:
        topology = inst["topology_type"]
        topology_values.append(topology)
        family = family_from_topology(topology)
        B = float(inst["B"])
        H = int(inst["H"])
        n_nodes = int(inst["n_nodes"])
        n_edges = int(inst["n_edges"])
        j = float(inst["J_star"])

        graph = inst.get("graph", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        is_directed = bool(graph.get("is_directed", True))
        directed_flags.append(is_directed)

        # Densité directionnelle (ou non directionnelle si non orienté)
        if n_nodes > 1:
            max_edges_dir = n_nodes * (n_nodes - 1)
            max_edges_undir = n_nodes * (n_nodes - 1) / 2
            if is_directed:
                density = n_edges / max_edges_dir
            else:
                density = n_edges / max_edges_undir
        else:
            density = 0.0

        # Existence d'un chemin source->target dans le graphe complet (tous noeuds UP)
        terminals = inst.get("terminals", [None, None])
        s, t = terminals[0], terminals[1]

        # BFS léger sans dépendance externe
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            if not is_directed:
                adj[v].append(u)

        reachable = False
        if s is not None and t is not None and s in set(nodes) and t in set(nodes):
            if s == t:
                reachable = True
            else:
                seen = {s}
                stack = [s]
                while stack:
                    cur = stack.pop()
                    for nxt in adj.get(cur, []):
                        if nxt == t:
                            reachable = True
                            stack.clear()
                            break
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)

        full_path_exists.append(reachable)

        alloc_metrics = allocation_quality_metrics(inst)
        alloc_s = alloc_metrics["allocation_spearman_centrality"]
        alloc_q = alloc_metrics["allocation_efficiency_score"]
        b_conc = alloc_metrics["budget_concentration"]
        b_active = bool(alloc_metrics["budget_active"])

        allocation_spearman_values.append(alloc_s)
        allocation_efficiency_values.append(alloc_q)
        budget_concentration_values.append(b_conc)
        budget_active_flags.append(b_active)

        influence_rows.append({
            "instance_index": len(influence_rows),
            "topology_type": topology,
            "family": family,
            "B": B,
            "H": H,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "J_star": j,
            "allocation_spearman_centrality": alloc_s,
            "allocation_efficiency_score": alloc_q,
            "budget_concentration": b_conc,
            "budget_active": b_active,
        })

        groups_topology[topology].append(j)
        groups_family[family].append(j)
        groups_H[H].append(j)
        groups_N[n_nodes].append(j)

        if B <= B_q25:
            b_bin = "low_B_q25"
        elif B >= B_q75:
            b_bin = "high_B_q75"
        else:
            b_bin = "mid_B"
        groups_B_bin[b_bin].append(j)

        if density < 0.2:
            d_bin = "sparse_<0.2"
        elif density < 0.4:
            d_bin = "medium_[0.2,0.4)"
        else:
            d_bin = "dense_>=0.4"
        groups_density_bin[d_bin].append(j)

        # Segments demandés: petit budget + petit réseau / H important + grand réseau
        small_budget = B <= B_q25
        large_budget = B >= B_q75
        small_network = n_nodes <= N_q25
        large_network = n_nodes >= N_q75
        high_H = H >= H_q75
        low_H = H <= H_q25

        if small_budget and small_network:
            groups_segment["small_budget_and_small_network"].append(j)
        if high_H and large_network:
            groups_segment["high_H_and_large_network"].append(j)
        if small_budget and large_network:
            groups_segment["small_budget_and_large_network"].append(j)
        if large_budget and large_network:
            groups_segment["large_budget_and_large_network"].append(j)
        if high_H and small_network:
            groups_segment["high_H_and_small_network"].append(j)
        if low_H and large_network:
            groups_segment["low_H_and_large_network"].append(j)

    overall = summarize_j(J_values)

    family_stats = {}
    for fam, vals in sorted(groups_family.items()):
        family_stats[fam] = summarize_j(vals)
        family_stats[fam]["failure_rate_ge_0_9"] = rate_above(vals, 0.9)
        family_stats[fam]["failure_rate_le_0_1"] = rate_below(vals, 0.1)

    topology_stats = {}
    for top, vals in sorted(groups_topology.items()):
        topology_stats[top] = {
            "count": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p50": float(np.percentile(vals, 50)),
            "p90": float(np.percentile(vals, 90)),
            "p10": float(np.percentile(vals, 10)),
            "failure_rate_ge_0_9": rate_above(vals, 0.9),
            "failure_rate_le_0.1": rate_below(vals, 0.1),
        }

    H_stats = {str(h): summarize_j(v) for h, v in sorted(groups_H.items())}
    N_stats = {str(n): summarize_j(v) for n, v in sorted(groups_N.items())}
    B_bin_stats = {k: summarize_j(v) for k, v in sorted(groups_B_bin.items())}
    density_stats = {k: summarize_j(v) for k, v in sorted(groups_density_bin.items())}

    segment_stats = {}
    for seg, vals in sorted(groups_segment.items()):
        if len(vals) == 0:
            continue
        seg_summary = summarize_j(vals)
        seg_summary["lift_vs_global_mean"] = float(seg_summary["mean"] - overall["mean"])
        seg_summary["failure_rate_ge_0_9"] = rate_above(vals, 0.9)
        seg_summary["failure_rate_le_0.1"] = rate_below(vals, 0.1)
        segment_stats[seg] = seg_summary

    # Classements topologies
    ranked_topology_mean = sorted(
        [(k, v["mean"], v["count"]) for k, v in topology_stats.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    correlations = {
        "corr(B, J_star)": safe_corr(B_values, J_values),
        "corr_spearman(B, J_star)": safe_spearman(B_values, J_values),
        "corr(H, J_star)": safe_corr(H_values, J_values),
        "corr(n_nodes, J_star)": safe_corr(N_values, J_values),
        "corr(n_edges, J_star)": safe_corr(E_values, J_values),
        "corr(allocation_spearman_centrality, J_star)": safe_corr(allocation_spearman_values, J_values),
        "corr(allocation_efficiency_score, J_star)": safe_corr(allocation_efficiency_values, J_values),
        "corr(budget_concentration, J_star)": safe_corr(budget_concentration_values, J_values),
    }

    topo_dummies, topo_levels, topo_baseline = onehot_topology(topology_values)
    controls_basic = np.column_stack([
        np.array(H_values, dtype=float),
        np.array(N_values, dtype=float),
        np.array(E_values, dtype=float),
    ])
    controls_budget_complexity = np.column_stack([
        np.array(B_values, dtype=float),
        np.array(H_values, dtype=float),
        np.array(N_values, dtype=float),
        np.array(E_values, dtype=float),
        topo_dummies,
    ])
    controls_with_alloc = np.column_stack([
        np.array(B_values, dtype=float),
        np.array(allocation_efficiency_values, dtype=float),
        np.array(H_values, dtype=float),
        np.array(N_values, dtype=float),
        np.array(E_values, dtype=float),
        topo_dummies,
    ])

    adjusted_budget_influence = {
        "raw_corr_pearson": safe_corr(B_values, J_values),
        "raw_corr_spearman": safe_spearman(B_values, J_values),
        "partial_corr_pearson__controls_H_n_nodes_n_edges": partial_corr_with_p(
            B_values, J_values, controls_basic
        ),
        "partial_corr_pearson__controls_H_n_nodes_n_edges_topology": partial_corr_with_p(
            B_values,
            J_values,
            np.column_stack([controls_basic, topo_dummies]),
        ),
        "standardized_beta_B__controls_H_n_nodes_n_edges": standardized_beta(
            J_values, B_values, controls_basic
        ),
        "standardized_beta_B__controls_H_n_nodes_n_edges_topology": standardized_beta(
            J_values,
            B_values,
            np.column_stack([controls_basic, topo_dummies]),
        ),
        "topology_encoding": {
            "levels": topo_levels,
            "baseline": topo_baseline,
        },
    }

    interaction_term = np.array(B_values, dtype=float) * np.array(allocation_efficiency_values, dtype=float)
    allocation_influence = {
        "raw_corr_allocation_spearman_vs_J_star": safe_corr(allocation_spearman_values, J_values),
        "raw_corr_allocation_efficiency_vs_J_star": safe_corr(allocation_efficiency_values, J_values),
        "partial_corr_allocation_spearman_vs_J_star__controls_B_H_n_nodes_n_edges_topology": partial_corr_with_p(
            allocation_spearman_values,
            J_values,
            controls_budget_complexity,
        ),
        "partial_corr_allocation_efficiency_vs_J_star__controls_B_H_n_nodes_n_edges_topology": partial_corr_with_p(
            allocation_efficiency_values,
            J_values,
            controls_budget_complexity,
        ),
        "partial_corr_interaction_Bxallocation_efficiency_vs_J_star": partial_corr_with_p(
            interaction_term,
            J_values,
            controls_with_alloc,
        ),
        "budget_active_ratio": float(np.mean(np.array(budget_active_flags, dtype=bool))),
        "allocation_efficiency_mean": float(np.mean(np.array(allocation_efficiency_values, dtype=float))),
        "allocation_efficiency_std": float(np.std(np.array(allocation_efficiency_values, dtype=float))),
    }

    alloc_q25 = percentile(allocation_efficiency_values, 25)
    alloc_q75 = percentile(allocation_efficiency_values, 75)
    j_arr = np.array(J_values, dtype=float)
    alloc_arr = np.array(allocation_efficiency_values, dtype=float)
    low_alloc_mask = alloc_arr <= alloc_q25
    high_alloc_mask = alloc_arr >= alloc_q75
    if np.any(low_alloc_mask) and np.any(high_alloc_mask):
        allocation_influence["J_star_mean_low_allocation_q25"] = float(np.mean(j_arr[low_alloc_mask]))
        allocation_influence["J_star_mean_high_allocation_q75"] = float(np.mean(j_arr[high_alloc_mask]))
        allocation_influence["J_star_lift_high_vs_low_allocation"] = float(
            np.mean(j_arr[high_alloc_mask]) - np.mean(j_arr[low_alloc_mask])
        )

    risk_bands = {
        "very_low_[0,0.1]": rate_below(J_values, 0.1),
        "low_(0.1,0.3]": float(np.mean((np.array(J_values) > 0.1) & (np.array(J_values) <= 0.3))),
        "medium_(0.3,0.6]": float(np.mean((np.array(J_values) > 0.3) & (np.array(J_values) <= 0.6))),
        "high_(0.6,0.9]": float(np.mean((np.array(J_values) > 0.6) & (np.array(J_values) <= 0.9))),
        "very_high_(0.9,1.0]": rate_above(J_values, 0.9),
    }

    analysis = {
        "dataset_path": str(dataset_path),
        "n_instances": len(instances),
        "metadata": data.get("metadata", {}),
        "thresholds": {
            "B_q25": B_q25,
            "B_q75": B_q75,
            "H_q25": H_q25,
            "H_q75": H_q75,
            "N_q25": N_q25,
            "N_q75": N_q75,
        },
        "overall_J_star": overall,
        "risk_bands": risk_bands,
        "correlations": correlations,
        "adjusted_budget_influence": adjusted_budget_influence,
        "allocation_influence": allocation_influence,
        "graph_integrity": {
            "directed_ratio": float(np.mean(np.array(directed_flags, dtype=bool))),
            "full_graph_path_exists_ratio": float(np.mean(np.array(full_path_exists, dtype=bool))),
        },
        "by_family": family_stats,
        "by_topology": topology_stats,
        "by_H": H_stats,
        "by_n_nodes": N_stats,
        "by_B_bin": B_bin_stats,
        "by_density_bin": density_stats,
        "key_segments": segment_stats,
        "topology_ranking_by_mean_J_desc": ranked_topology_mean,
        "topology_highest_5": ranked_topology_mean[:5],
        "topology_lowest_5": ranked_topology_mean[-5:],
    }

    with out_json.open("w") as f:
        json.dump(analysis, f, indent=2)

    influence_payload = {
        "dataset_path": str(dataset_path),
        "n_instances": len(instances),
        "variables": {
            "adjusted_budget_influence": adjusted_budget_influence,
            "allocation_influence": allocation_influence,
            "thresholds": {
                "allocation_efficiency_q25": alloc_q25,
                "allocation_efficiency_q75": alloc_q75,
            },
        },
        "instances": influence_rows,
    }
    with out_influence_json.open("w") as f:
        json.dump(influence_payload, f, indent=2)

    with out_influence_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "instance_index",
            "topology_type",
            "family",
            "B",
            "H",
            "n_nodes",
            "n_edges",
            "J_star",
            "allocation_spearman_centrality",
            "allocation_efficiency_score",
            "budget_concentration",
            "budget_active",
        ])
        writer.writeheader()
        writer.writerows(influence_rows)

    lines = []
    lines.append(f"ANALYSE DATASET - {dataset_path.name} - PROBABILITÉS DE DÉFAILLANCE\n")
    lines.append(f"Instances: {analysis['n_instances']}")
    lines.append(f"J* mean={overall['mean']:.6f} | median={overall['p50']:.6f} | p90={overall['p90']:.6f} | p99={overall['p99']:.6f}")
    lines.append("\nBandes de risque (part du dataset):")
    for k, v in risk_bands.items():
        lines.append(f"- {k}: {v:.2%}")

    lines.append("\nCorrélations avec J*:")
    for k, v in correlations.items():
        lines.append(f"- {k}: {v if v is not None else 'NA'}")

    lines.append("\nInfluence réelle du budget (corrigée de la complexité):")
    for k, v in adjusted_budget_influence.items():
        if isinstance(v, dict):
            lines.append(f"- {k}: {json.dumps(v)}")
        else:
            lines.append(f"- {k}: {v}")

    lines.append("\nInfluence de la qualité de répartition du budget:")
    for k, v in allocation_influence.items():
        if isinstance(v, dict):
            lines.append(f"- {k}: {json.dumps(v)}")
        else:
            lines.append(f"- {k}: {v}")

    lines.append("\nFamilles de topologies (mean J*):")
    for fam, stats in sorted(family_stats.items(), key=lambda x: x[1]['mean'], reverse=True):
        lines.append(
            f"- {fam}: count={stats['count']}, mean={stats['mean']:.6f}, p50={stats['p50']:.6f}, p90={stats['p90']:.6f}, P(J*>=0.9)={stats['failure_rate_ge_0_9']:.2%}"
        )

    lines.append("\nSegments clés:")
    for seg, stats in sorted(segment_stats.items(), key=lambda x: x[1]['mean'], reverse=True):
        lines.append(
            f"- {seg}: count={stats['count']}, mean={stats['mean']:.6f}, lift_vs_global={stats['lift_vs_global_mean']:+.6f}, P(J*>=0.9)={stats['failure_rate_ge_0_9']:.2%}"
        )

    lines.append("\nTop 5 topologies les plus risquées (J* moyen):")
    for top, mean_j, count in analysis["topology_highest_5"]:
        lines.append(f"- {top}: mean={mean_j:.6f} (count={count})")

    lines.append("\nTop 5 topologies les moins risquées (J* moyen):")
    for top, mean_j, count in analysis["topology_lowest_5"]:
        lines.append(f"- {top}: mean={mean_j:.6f} (count={count})")

    out_txt.write_text("\n".join(lines))

    print(
        "Analyse terminée.\n"
        f"- JSON: {out_json}\n"
        f"- Résumé: {out_txt}\n"
        f"- Variables d'influence (JSON): {out_influence_json}\n"
        f"- Variables d'influence (CSV): {out_influence_csv}"
    )


if __name__ == "__main__":
    main()
