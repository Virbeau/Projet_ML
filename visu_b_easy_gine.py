import argparse
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import networkx as nx
import numpy as np
import torch

from benchmark_b_no_solver_complex import (
    _load_gine_b,
    _predict_gine_b,
    _simulate_mc_with_policy,
    naive_greedy_utility,
    naive_risk_prop,
    naive_uniform,
)


EASY_PROFILE_PARAMS = {
    "fragile": {"p": (0.06, 0.12), "c": (1.0, 2.6)},
    "stable": {"p": (0.02, 0.06), "c": (2.6, 5.2)},
    "relay": {"p": (0.04, 0.09), "c": (1.4, 3.4)},
    "bottleneck": {"p": (0.08, 0.14), "c": (1.8, 4.5)},
}

FAMILY_COLORS = {
    "parallel": "#1E88E5",
    "ladder": "#43A047",
    "grid": "#FB8C00",
    "diamond": "#8E24AA",
}


def _sample_profile(rng, centrality_rank, n_rep):
    if centrality_rank < max(1, n_rep // 5):
        return "bottleneck"
    profiles = ["fragile", "stable", "relay"]
    weights = [0.35, 0.25, 0.40]
    return rng.choices(profiles, weights=weights, k=1)[0]


def _build_instance(graph, positions, family, horizon, seed):
    rng = random.Random(seed)
    terminals = [n for n, data in graph.nodes(data=True) if data.get("terminal")]
    source = [n for n in terminals if graph.nodes[n].get("terminal") == "source"][0]
    target = [n for n in terminals if graph.nodes[n].get("terminal") == "target"][0]
    repairable_nodes = [n for n in graph.nodes() if n not in (source, target)]

    centrality = nx.betweenness_centrality(graph.to_undirected())
    ranked_nodes = sorted(repairable_nodes, key=lambda node: centrality[node], reverse=True)
    profile_map = {}
    for rank, node in enumerate(ranked_nodes):
        profile_map[node] = _sample_profile(rng, rank, len(repairable_nodes))

    node_features = {}
    for node in graph.nodes():
        if node in (source, target):
            p_fail = 0.0
            c_cost = 0.0
        else:
            profile = profile_map[node]
            params = EASY_PROFILE_PARAMS[profile]
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
    alpha = float(np.clip(rng.gauss(0.14 + 0.48 * (mean_risk - 0.05), 0.020), 0.09, 0.22))
    budget = round(alpha * max(c_total, 1.0), 2)

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
        "topology_type": f"easy_{family}",
        "family": family,
        "graph": {
            "nodes": sorted(graph.nodes()),
            "edges": list(graph.edges()),
            "is_directed": True,
        },
        "terminals": [source, target],
        "repairable_nodes": repairable_nodes,
        "x": x,
        "y": [0.0 for _ in sorted(graph.nodes())],
        "B": float(budget),
        "C_total": float(c_total),
        "H": int(horizon),
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "viz_pos": {str(node): [float(positions[node][0]), float(positions[node][1])] for node in positions},
        "seed": int(seed),
    }


def _generate_parallel_family(seed, horizon):
    graph = nx.DiGraph()
    source, target = 0, 1
    graph.add_node(source, terminal="source")
    graph.add_node(target, terminal="target")
    positions = {source: (0.0, 0.0), target: (4.0, 0.0)}
    next_id = 2
    lanes = []
    lane_ys = [1.3, 0.0, -1.3]
    for lane_idx, y in enumerate(lane_ys):
        lane_nodes = []
        for x in [1.0, 2.0, 3.0]:
            node = next_id
            next_id += 1
            graph.add_node(node)
            lane_nodes.append(node)
            positions[node] = (x, y)
        lanes.append(lane_nodes)
        graph.add_edge(source, lane_nodes[0])
        graph.add_edge(lane_nodes[0], lane_nodes[1])
        graph.add_edge(lane_nodes[1], lane_nodes[2])
        graph.add_edge(lane_nodes[2], target)
    for left, right in zip(lanes[:-1], lanes[1:]):
        graph.add_edge(left[1], right[1])
        graph.add_edge(right[1], left[1])
        graph.add_edge(left[2], right[2])
    return _build_instance(graph, positions, "parallel", horizon, seed)


def _generate_ladder_family(seed, horizon):
    graph = nx.DiGraph()
    source, target = 0, 1
    graph.add_node(source, terminal="source")
    graph.add_node(target, terminal="target")
    positions = {source: (0.0, 0.0), target: (5.0, 0.0)}
    next_id = 2
    top_nodes, bottom_nodes = [], []
    for col in range(1, 5):
        top = next_id
        bottom = next_id + 1
        next_id += 2
        graph.add_node(top)
        graph.add_node(bottom)
        top_nodes.append(top)
        bottom_nodes.append(bottom)
        positions[top] = (float(col), 0.9)
        positions[bottom] = (float(col), -0.9)

    graph.add_edge(source, top_nodes[0])
    graph.add_edge(source, bottom_nodes[0])
    graph.add_edge(top_nodes[-1], target)
    graph.add_edge(bottom_nodes[-1], target)
    for first, second in zip(top_nodes[:-1], top_nodes[1:]):
        graph.add_edge(first, second)
    for first, second in zip(bottom_nodes[:-1], bottom_nodes[1:]):
        graph.add_edge(first, second)
    for top, bottom in zip(top_nodes, bottom_nodes):
        graph.add_edge(top, bottom)
        graph.add_edge(bottom, top)
    for first, second in zip(top_nodes[:-1], bottom_nodes[1:]):
        graph.add_edge(first, second)
    for first, second in zip(bottom_nodes[:-1], top_nodes[1:]):
        graph.add_edge(first, second)
    return _build_instance(graph, positions, "ladder", horizon, seed)


def _generate_grid_family(seed, horizon):
    graph = nx.DiGraph()
    positions = {}
    rows, cols = 3, 4
    node_ids = {}
    next_id = 0
    for row in range(rows):
        for col in range(cols):
            node_ids[(row, col)] = next_id
            terminal = None
            if row == 0 and col == 0:
                terminal = "source"
            if row == rows - 1 and col == cols - 1:
                terminal = "target"
            if terminal is None:
                graph.add_node(next_id)
            else:
                graph.add_node(next_id, terminal=terminal)
            positions[next_id] = (float(col), float(-row))
            next_id += 1
    for row in range(rows):
        for col in range(cols):
            current = node_ids[(row, col)]
            if col + 1 < cols:
                graph.add_edge(current, node_ids[(row, col + 1)])
            if row + 1 < rows:
                graph.add_edge(current, node_ids[(row + 1, col)])
            if row + 1 < rows and col + 1 < cols:
                graph.add_edge(current, node_ids[(row + 1, col + 1)])
    return _build_instance(graph, positions, "grid", horizon, seed)


def _generate_diamond_family(seed, horizon):
    graph = nx.DiGraph()
    source, target = 0, 1
    graph.add_node(source, terminal="source")
    graph.add_node(target, terminal="target")
    positions = {source: (0.0, 0.0), target: (6.0, 0.0)}
    next_id = 2
    entry = source
    x_pos = 1.0
    for stage in range(2):
        up = next_id
        mid = next_id + 1
        down = next_id + 2
        merge = next_id + 3
        next_id += 4
        for node in [up, mid, down, merge]:
            graph.add_node(node)
        positions[up] = (x_pos, 1.1)
        positions[mid] = (x_pos, 0.0)
        positions[down] = (x_pos, -1.1)
        positions[merge] = (x_pos + 1.0, 0.0)
        graph.add_edge(entry, up)
        graph.add_edge(entry, mid)
        graph.add_edge(entry, down)
        graph.add_edge(entry, merge)
        graph.add_edge(up, merge)
        graph.add_edge(mid, merge)
        graph.add_edge(down, merge)
        graph.add_edge(up, mid)
        graph.add_edge(mid, down)
        graph.add_edge(up, down)
        entry = merge
        x_pos += 2.0
    graph.add_edge(entry, target)
    return _build_instance(graph, positions, "diamond", horizon, seed)


def _generate_easy_instance(family, seed, horizon):
    builders = {
        "parallel": _generate_parallel_family,
        "ladder": _generate_ladder_family,
        "grid": _generate_grid_family,
        "diamond": _generate_diamond_family,
    }
    return builders[family](seed, horizon)


def _alloc_array(inst, pi_dict):
    nodes = inst["graph"]["nodes"]
    return np.array([float(pi_dict.get(node, 0.0)) for node in nodes], dtype=float)


def _draw_graph(ax, inst, alloc_values, title, cmap, norm):
    graph = nx.DiGraph()
    graph.add_nodes_from(inst["graph"]["nodes"])
    graph.add_edges_from(inst["graph"]["edges"])
    positions = {int(node): tuple(pos) for node, pos in inst["viz_pos"].items()}
    nodes = inst["graph"]["nodes"]
    source, target = inst["terminals"]
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        arrows=False,
        edge_color="#B0BEC5",
        alpha=0.35,
        width=1.0,
    )

    rep_nodes = [node for node in nodes if node not in (source, target)]
    node_colors = [cmap(norm(float(alloc_values[node_to_idx[node]]))) for node in rep_nodes]
    node_sizes = [350 + 1200 * float(alloc_values[node_to_idx[node]]) for node in rep_nodes]
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=rep_nodes,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=1.1,
        edgecolors="#37474F",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[source],
        node_color="#1E88E5",
        node_size=720,
        linewidths=1.4,
        edgecolors="#0D47A1",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[target],
        node_color="#43A047",
        node_size=720,
        linewidths=1.4,
        edgecolors="#1B5E20",
        ax=ax,
    )
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(
        description="Évaluer GINE_B sur graphes simples, redondants et visuellement lisibles"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/gine_b_repartition_v7.pt")
    parser.add_argument("--instances-per-family", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--mc-sims", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-json", type=str,
                        default="visu_b_easy_gine_results.json")
    parser.add_argument("--out-summary", type=str,
                        default="visu_b_easy_gine.png")
    parser.add_argument("--out-instance", type=str,
                        default="visu_b_easy_gine_instance.png")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    model = _load_gine_b(args.checkpoint, device)

    families = ["parallel", "ladder", "grid", "diamond"]
    rows = []
    exemplars = {}

    print(
        f"Benchmark visuel GINE_B | familles={families} | "
        f"instances/famille={args.instances_per_family} | H={args.horizon} | mc_sims={args.mc_sims}"
    )

    for family_idx, family in enumerate(families):
        for local_idx in range(args.instances_per_family):
            inst_seed = args.seed + 1000 * family_idx + 73 * local_idx
            inst = _generate_easy_instance(family, inst_seed, args.horizon)
            pi_gine, infer_ms = _predict_gine_b(model, inst, device)
            alloc_gine = _alloc_array(inst, pi_gine)

            pi_uniform = naive_uniform(inst)
            pi_risk = naive_risk_prop(inst)
            pi_greedy = naive_greedy_utility(inst)

            mc_seed = inst_seed + 999
            j_uniform = _simulate_mc_with_policy(inst, pi_uniform, args.mc_sims, mc_seed)
            j_risk = _simulate_mc_with_policy(inst, pi_risk, args.mc_sims, mc_seed + 1)
            j_greedy = _simulate_mc_with_policy(inst, pi_greedy, args.mc_sims, mc_seed + 2)
            j_gine = _simulate_mc_with_policy(inst, pi_gine, args.mc_sims, mc_seed + 3)
            best_naive = min(j_uniform, j_risk, j_greedy)
            delta_gain = best_naive - j_gine

            row = {
                "family": family,
                "seed": inst_seed,
                "n_nodes": inst["n_nodes"],
                "n_edges": inst["n_edges"],
                "B": inst["B"],
                "C_total": inst["C_total"],
                "budget_ratio": float(inst["B"] / max(inst["C_total"], 1e-8)),
                "J_uniform": float(j_uniform),
                "J_risk_prop": float(j_risk),
                "J_greedy": float(j_greedy),
                "J_gine_b": float(j_gine),
                "best_naive": float(best_naive),
                "delta_gain": float(delta_gain),
                "gine_inference_ms": float(infer_ms),
                "instance": inst,
                "alloc_gine": alloc_gine.tolist(),
            }
            rows.append(row)
            if family not in exemplars:
                exemplars[family] = row

            print(
                f"[{family:<8} #{local_idx}] n={inst['n_nodes']:>2} | "
                f"B/C={row['budget_ratio']:.2f} | "
                f"J_best_naive={best_naive:.4f} | J_GINE={j_gine:.4f} | ΔJ={delta_gain:+.4f}"
            )

    summary = {
        "config": {
            "checkpoint": args.checkpoint,
            "instances_per_family": args.instances_per_family,
            "horizon": args.horizon,
            "mc_sims": args.mc_sims,
            "seed": args.seed,
            "families": families,
        },
        "rows": rows,
    }
    with open(args.out_json, "w") as handle:
        json.dump(summary, handle, indent=2)

    all_best = np.array([row["best_naive"] for row in rows], dtype=float)
    all_gine = np.array([row["J_gine_b"] for row in rows], dtype=float)
    all_gain = np.array([row["delta_gain"] for row in rows], dtype=float)
    budget_ratios = np.array([row["budget_ratio"] for row in rows], dtype=float)

    fig = plt.figure(figsize=(17.0, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0, 1.1], hspace=0.28, wspace=0.22)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_gain = fig.add_subplot(gs[0, 1])
    ax_parallel = fig.add_subplot(gs[0, 2])
    ax_ladder = fig.add_subplot(gs[1, 0])
    ax_grid = fig.add_subplot(gs[1, 1])
    ax_diamond = fig.add_subplot(gs[1, 2])

    min_val = float(min(np.min(all_best), np.min(all_gine)))
    max_val = float(max(np.max(all_best), np.max(all_gine)))
    pad = 0.02 * max(1.0, max_val - min_val)
    for family in families:
        fam_rows = [row for row in rows if row["family"] == family]
        ax_scatter.scatter(
            [row["best_naive"] for row in fam_rows],
            [row["J_gine_b"] for row in fam_rows],
            s=[120 + 320 * row["budget_ratio"] for row in fam_rows],
            color=FAMILY_COLORS[family],
            alpha=0.88,
            edgecolors="#263238",
            linewidths=0.6,
            label=family,
        )
    ax_scatter.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad],
                    linestyle="--", color="#D32F2F", linewidth=1.1)
    ax_scatter.set_xlim(min_val - pad, max_val + pad)
    ax_scatter.set_ylim(min_val - pad, max_val + pad)
    ax_scatter.set_xlabel("Meilleur J naïf")
    ax_scatter.set_ylabel("J de GINE_B")
    ax_scatter.set_title("Généralisation sur graphes simples\nSous la diagonale = GINE_B meilleur", fontsize=10, fontweight="bold")
    ax_scatter.grid(alpha=0.22)
    ax_scatter.legend(fontsize=8)

    family_to_x = {family: idx for idx, family in enumerate(families)}
    for family in families:
        fam_gain = np.array([row["delta_gain"] for row in rows if row["family"] == family], dtype=float)
        x_values = np.full_like(fam_gain, family_to_x[family], dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(fam_gain)) if len(fam_gain) > 1 else np.array([0.0])
        ax_gain.scatter(x_values + jitter, fam_gain, color=FAMILY_COLORS[family], s=90,
                        alpha=0.9, edgecolors="#263238", linewidths=0.5)
        ax_gain.hlines(float(np.mean(fam_gain)), family_to_x[family] - 0.20, family_to_x[family] + 0.20,
                       colors="#263238", linewidth=2.0)
    ax_gain.axhline(0.0, color="#D32F2F", linestyle="--", linewidth=1.1)
    ax_gain.set_xticks(range(len(families)))
    ax_gain.set_xticklabels(families)
    ax_gain.set_ylabel("ΔJ = J_best_naive - J_GINE_B")
    ax_gain.set_title(
        f"Gain par famille\nMoyenne globale ΔJ={float(np.mean(all_gain)):+.4f} | B/C moyen={float(np.mean(budget_ratios)):.2f}",
        fontsize=10,
        fontweight="bold",
    )
    ax_gain.grid(alpha=0.22, axis="y")

    cmap = plt.get_cmap("YlOrRd")
    max_alloc = max(float(np.max(row["alloc_gine"])) for row in exemplars.values())
    norm = Normalize(vmin=0.0, vmax=max(0.6, max_alloc))
    for axis, family in zip([ax_parallel, ax_ladder, ax_grid, ax_diamond], families):
        row = exemplars[family]
        inst = row["instance"]
        gain_label = f"ΔJ={row['delta_gain']:+.4f}"
        _draw_graph(
            axis,
            inst,
            np.array(row["alloc_gine"], dtype=float),
            f"{family} | J_GINE={row['J_gine_b']:.4f} | best={row['best_naive']:.4f} | {gain_label}",
            cmap,
            norm,
        )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_parallel, ax_ladder, ax_grid, ax_diamond], fraction=0.022, pad=0.015)
    cbar.set_label("π alloué par GINE_B")

    win_rate = 100.0 * float(np.mean(all_gain > 0))
    easy_j_mean = float(np.mean(all_gine))
    fig.suptitle(
        f"GINE_B sur topologies simples et redondantes | win-rate={win_rate:.1f}% | J moyen={easy_j_mean:.4f}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(args.out_summary, dpi=args.dpi, bbox_inches="tight", facecolor="white")

    first_row = rows[0]
    fig_instance = plt.figure(figsize=(7.0, 5.4))
    ax_instance = fig_instance.add_subplot(111)
    _draw_graph(
        ax_instance,
        first_row["instance"],
        np.array(first_row["alloc_gine"], dtype=float),
        (
            f"Première instance facile | {first_row['family']}\n"
            f"J_GINE={first_row['J_gine_b']:.4f} | best_naive={first_row['best_naive']:.4f} | ΔJ={first_row['delta_gain']:+.4f}"
        ),
        cmap,
        norm,
    )
    cbar_instance = fig_instance.colorbar(sm, ax=ax_instance, fraction=0.046, pad=0.03)
    cbar_instance.set_label("π alloué par GINE_B")
    fig_instance.savefig(args.out_instance, dpi=args.dpi, bbox_inches="tight", facecolor="white")

    print(f"\nJSON sauvegardé : {args.out_json}")
    print(f"Figure générale : {args.out_summary}")
    print(f"Figure première instance : {args.out_instance}")
    print(f"J moyen GINE_B : {easy_j_mean:.4f}")
    print(f"Gain moyen ΔJ : {float(np.mean(all_gain)):+.4f}")
    print(f"Budget relatif moyen B/C : {float(np.mean(budget_ratios)):.2f}")


if __name__ == "__main__":
    main()