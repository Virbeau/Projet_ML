#!/usr/bin/env python3
import json
import math
from pathlib import Path
from collections import defaultdict
import statistics
import argparse

import numpy as np


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

    for inst in instances:
        topology = inst["topology_type"]
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
        "corr(H, J_star)": safe_corr(H_values, J_values),
        "corr(n_nodes, J_star)": safe_corr(N_values, J_values),
        "corr(n_edges, J_star)": safe_corr(E_values, J_values),
    }

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

    print(f"Analyse terminée.\n- JSON: {out_json}\n- Résumé: {out_txt}")


if __name__ == "__main__":
    main()
