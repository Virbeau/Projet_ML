import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import networkx as nx
import numpy as np
import torch

from benchmark_b_no_solver_complex import _load_gine_b, _predict_gine_b
from benchmark_j_no_solver_complex import generate_complex_instance


def _load_results(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _best_naive(row):
    return min(row["J_uniform"], row["J_risk_prop"], row["J_greedy"])


def _resolve_instance(results, row, instance_index):
    if row.get("instance") is not None:
        return row["instance"]

    cfg = results["config"]
    return generate_complex_instance(
        cfg["min_nodes"],
        cfg["max_nodes"],
        cfg["horizon"],
        row["seed"],
    )


def _alloc_array(inst, pi_dict):
    nodes = inst["graph"]["nodes"]
    return np.array([float(pi_dict.get(n, 0.0)) for n in nodes], dtype=float)


def _graph_layout(inst):
    graph = nx.DiGraph()
    graph.add_nodes_from(inst["graph"]["nodes"])
    graph.add_edges_from(inst["graph"]["edges"])
    return graph, nx.spring_layout(graph, seed=17, k=1.4 / max(1, np.sqrt(len(graph.nodes()))))


def _draw_instance_allocation(ax, inst, alloc_values, title, cmap, norm):
    graph, pos = _graph_layout(inst)
    nodes = inst["graph"]["nodes"]
    source, target = inst["terminals"]
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color="#B0BEC5",
        alpha=0.18,
        arrows=False,
        width=0.8,
    )

    repairable_nodes = [n for n in nodes if n not in (source, target)]
    repairable_colors = [cmap(norm(float(alloc_values[node_to_idx[n]]))) for n in repairable_nodes]
    repairable_sizes = [260 + 900 * float(alloc_values[node_to_idx[n]]) for n in repairable_nodes]

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=repairable_nodes,
        node_color=repairable_colors,
        node_size=repairable_sizes,
        linewidths=0.9,
        edgecolors="#37474F",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=[source],
        node_color="#1E88E5",
        node_size=650,
        linewidths=1.4,
        edgecolors="#0D47A1",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=[target],
        node_color="#43A047",
        node_size=650,
        linewidths=1.4,
        edgecolors="#1B5E20",
        ax=ax,
    )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(
        description="Visualisation compacte de la généralisation de GINE_B"
    )
    parser.add_argument("--results-json", type=str,
                        default="benchmark_b_complex_gine.json",
                        help="JSON de résultats du benchmark B")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint GINE_B; défaut = celui du JSON")
    parser.add_argument("--instance-index", type=int, default=0,
                        help="Index de l'instance à afficher")
    parser.add_argument("--out", type=str,
                        default="visu_b_gen_gine.png",
                        help="Image récapitulative")
    parser.add_argument("--out-instance", type=str,
                        default="visu_b_gen_gine_instance.png",
                        help="Image dédiée à l'instance affichée")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    results = _load_results(args.results_json)
    rows = results["rows"]
    if not rows:
        raise ValueError("Aucune ligne dans le JSON de benchmark")

    if args.instance_index < 0 or args.instance_index >= len(rows):
        raise ValueError("--instance-index hors bornes")

    best_naive = np.array([_best_naive(row) for row in rows], dtype=float)
    j_gine = np.array([row["J_gine_b"] for row in rows], dtype=float)
    delta_gain = best_naive - j_gine
    n_nodes = np.array([row["n_nodes"] for row in rows], dtype=float)
    budgets = np.array([row["B"] for row in rows], dtype=float)
    win_rate = 100.0 * float(np.mean(delta_gain > 0))

    ref_row = rows[args.instance_index]
    ref_inst = _resolve_instance(results, ref_row, args.instance_index)

    checkpoint = args.checkpoint or results["config"]["checkpoint"]
    device = torch.device("cpu")
    model = _load_gine_b(checkpoint, device)
    pi_gine_dict, _ = _predict_gine_b(model, ref_inst, device)
    alloc_values = _alloc_array(ref_inst, pi_gine_dict)

    cmap = plt.get_cmap("YlOrRd")
    norm = Normalize(vmin=0.0, vmax=max(1.0, float(np.max(alloc_values)) if len(alloc_values) else 1.0))

    fig = plt.figure(figsize=(16.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.1], wspace=0.28)

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_graph = fig.add_subplot(gs[0, 2])

    point_sizes = 60 + 1.7 * (budgets - float(np.min(budgets)))
    scatter = ax_scatter.scatter(
        best_naive,
        j_gine,
        c=n_nodes,
        s=point_sizes,
        cmap="viridis",
        alpha=0.88,
        edgecolors="#263238",
        linewidths=0.6,
    )
    min_val = float(min(np.min(best_naive), np.min(j_gine)))
    max_val = float(max(np.max(best_naive), np.max(j_gine)))
    pad = 0.01 * max(1.0, max_val - min_val)
    ax_scatter.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad],
                    linestyle="--", color="#D32F2F", linewidth=1.1)
    ax_scatter.set_xlim(min_val - pad, max_val + pad)
    ax_scatter.set_ylim(min_val - pad, max_val + pad)
    ax_scatter.set_xlabel("Meilleur J naïf")
    ax_scatter.set_ylabel("J de GINE_B")
    ax_scatter.set_title(
        f"Généralisation instance par instance\nSous la diagonale = GINE_B meilleur | win-rate={win_rate:.1f}%",
        fontsize=10,
        fontweight="bold",
    )
    ax_scatter.grid(alpha=0.22)
    colorbar_scatter = fig.colorbar(scatter, ax=ax_scatter, fraction=0.046, pad=0.04)
    colorbar_scatter.set_label("Nombre de nœuds")

    bins = np.linspace(float(np.min(delta_gain)) - 0.005, float(np.max(delta_gain)) + 0.005, 16)
    ax_hist.hist(delta_gain, bins=bins, color="#FB8C00", edgecolor="white", alpha=0.9)
    ax_hist.axvline(0.0, color="#D32F2F", linestyle="--", linewidth=1.2)
    ax_hist.axvline(float(np.mean(delta_gain)), color="#263238", linestyle="-", linewidth=1.1)
    ax_hist.set_xlabel("ΔJ = J_best_naive - J_GINE_B")
    ax_hist.set_ylabel("Nombre d'instances")
    ax_hist.set_title(
        f"Distribution des gains\nMoyenne ΔJ={float(np.mean(delta_gain)):+.4f}",
        fontsize=10,
        fontweight="bold",
    )
    ax_hist.grid(alpha=0.22, axis="y")

    best_ref_naive = _best_naive(ref_row)
    delta_ref = best_ref_naive - ref_row["J_gine_b"]
    ref_status = "gagne" if delta_ref > 0 else "perd"
    _draw_instance_allocation(
        ax_graph,
        ref_inst,
        alloc_values,
        (
            f"Instance {args.instance_index} | allocation GINE_B\n"
            f"n={ref_row['n_nodes']} | B={ref_row['B']:.2f} | ΔJ={delta_ref:+.4f} ({ref_status})"
        ),
        cmap,
        norm,
    )
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar_graph = fig.colorbar(sm, ax=ax_graph, fraction=0.046, pad=0.03)
    colorbar_graph.set_label("π alloué par GINE_B")

    fig.suptitle(
        "Diagnostic rapide de généralisation de GINE_B",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")

    fig_instance = plt.figure(figsize=(6.4, 5.8))
    ax_instance = fig_instance.add_subplot(111)
    _draw_instance_allocation(
        ax_instance,
        ref_inst,
        alloc_values,
        (
            f"Première instance benchmark | GINE_B\n"
            f"J_GINE={ref_row['J_gine_b']:.4f} | meilleur naïf={best_ref_naive:.4f} | ΔJ={delta_ref:+.4f}"
        ),
        cmap,
        norm,
    )
    colorbar_instance = fig_instance.colorbar(sm, ax=ax_instance, fraction=0.046, pad=0.03)
    colorbar_instance.set_label("π alloué par GINE_B")
    fig_instance.savefig(args.out_instance, dpi=args.dpi, bbox_inches="tight", facecolor="white")

    print(f"Figure générale sauvegardée : {args.out}")
    print(f"Figure instance sauvegardée : {args.out_instance}")
    print(f"Win-rate vs meilleur naïf : {win_rate:.1f}%")
    print(f"Gain moyen ΔJ : {float(np.mean(delta_gain)):+.4f}")
    print(
        f"Instance {args.instance_index}: J_GINE={ref_row['J_gine_b']:.4f} | "
        f"best_naive={best_ref_naive:.4f} | ΔJ={delta_ref:+.4f}"
    )


if __name__ == "__main__":
    main()