#!/usr/bin/env python3
"""
Affiche un graphe tiré aléatoirement dans un dataset JSON.
Pour chaque nœud, tous les features x et y sont affichés.
Les métadonnées globales (B, H, J*, etc.) sont résumées dans un encart.

Usage:
    python3 inspect_instance.py
    python3 inspect_instance.py --dataset JSON/datasetV6.json --index 42
    python3 inspect_instance.py --dataset JSON/datasetV6.json --seed 7
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ------------------------------------------------------------------
# Noms des features de x (dans l'ordre de la liste)
# ------------------------------------------------------------------
X_FEATURE_NAMES = [
    "p_fail",
    "c_cost",
    "is_source",
    "is_target",
    "in_degree",
    "out_degree",
    "dist_to_target",
    "B",
    "H",
]


def load_dataset(path: str):
    with open(path) as f:
        data = json.load(f)
    instances = data.get("instances", [])
    if not instances:
        raise ValueError(f"Aucune instance trouvée dans {path}")
    return instances


def pick_instance(instances, index=None, seed=None):
    if index is not None:
        if index < 0 or index >= len(instances):
            raise ValueError(f"Index {index} hors plage [0, {len(instances)-1}]")
        return index, instances[index]
    rng = random.Random(seed)
    idx = rng.randint(0, len(instances) - 1)
    return idx, instances[idx]


def build_graph(inst):
    is_directed = bool(inst.get("graph", {}).get("is_directed", True))
    g = nx.DiGraph() if is_directed else nx.Graph()
    nodes = inst["graph"].get("nodes", list(range(inst["n_nodes"])))
    g.add_nodes_from(nodes)
    g.add_edges_from(inst["graph"].get("edges", []))
    return g, is_directed


def node_label_multiline(node_id, inst):
    """Construit le label multi-ligne positionné sur le nœud."""
    nodes = inst["graph"].get("nodes", list(range(inst["n_nodes"])))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx = node_to_idx.get(node_id)
    if idx is None:
        return str(node_id)

    x_vec = inst["x"][idx]
    y_val = inst["y"][idx]
    terminals = inst.get("terminals", [])
    repairable = inst.get("repairable_nodes", [])

    role_parts = []
    if node_id == terminals[0]:
        role_parts.append("SOURCE")
    if len(terminals) > 1 and node_id == terminals[1]:
        role_parts.append("TARGET")
    if node_id in repairable:
        role_parts.append("réparable")
    role = "/".join(role_parts) if role_parts else "non-réparable"

    lines = [f"N{node_id} [{role}]"]
    for name, val in zip(X_FEATURE_NAMES, x_vec):
        if name in ("B", "H"):          # globaux, pas utile de répéter sur chaque nœud
            continue
        if name in ("is_source", "is_target"):
            lines.append(f"{name}: {int(val)}")
        else:
            lines.append(f"{name}: {val:.3f}")
    lines.append(f"π* (y): {y_val:.4f}")
    return "\n".join(lines)


def node_colors_and_sizes(inst, g):
    """
    Couleur par rôle :
      - Source    : rouge fixe  (#e74c3c)
      - Target    : orange fixe (#e67e22)
      - Réparable : colormap YlOrRd selon budget alloué π*·c
      - Autre     : gris        (#bdc3c7)
    Retourne (colors, sizes, budget_per_node, max_budget_rep).
    """
    nodes = inst["graph"].get("nodes", list(range(inst["n_nodes"])))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    terminals = inst.get("terminals", [])
    repairable = inst.get("repairable_nodes", [])
    y = inst["y"]
    x = inst["x"]

    # Budget alloué par nœud = π* × c_cost
    budget_per_node = {}
    rep_budgets = []
    for nd in nodes:
        nd_idx = node_to_idx[nd]
        bval = float(y[nd_idx]) * float(x[nd_idx][1])
        budget_per_node[nd] = bval
        if nd in repairable:
            rep_budgets.append(bval)

    max_budget_rep = max(rep_budgets) if rep_budgets else 1.0
    cmap = cm.YlOrRd

    colors = []
    sizes = []
    for node in g.nodes():
        if node == terminals[0]:
            colors.append("#e74c3c")     # rouge = source
            sizes.append(1000)
        elif len(terminals) > 1 and node == terminals[1]:
            colors.append("#e67e22")     # orange = target
            sizes.append(1000)
        elif node in repairable:
            t = budget_per_node[node] / max_budget_rep if max_budget_rep > 0 else 0.0
            rgba = cmap(0.15 + t * 0.85)   # évite le blanc pur à t=0
            colors.append(mcolors.to_hex(rgba))
            sizes.append(800)
        else:
            colors.append("#bdc3c7")     # gris = non-réparable, non-terminal
            sizes.append(600)

    return colors, sizes, budget_per_node, max_budget_rep


def draw_instance(inst, idx_in_dataset: int, output_path: str):
    g, is_directed = build_graph(inst)
    n = inst["n_nodes"]

    # Layout: spring pour petit graphe, kamada_kawai pour plus grand
    pos = (nx.kamada_kawai_layout(g) if n <= 20
           else nx.spring_layout(g, seed=42, k=2.5 / max(1, n ** 0.5)))

    colors, sizes, budget_per_node_alloc, max_budget_rep = node_colors_and_sizes(inst, g)

    # Calcul du budget effectivement dépensé (somme pi_i * c_i)
    nodes_list = inst["graph"].get("nodes", list(range(n)))
    node_to_idx = {nd: i for i, nd in enumerate(nodes_list)}
    B = float(inst["B"])
    c_cost_vec = np.array([float(inst["x"][node_to_idx[nd]][1]) for nd in nodes_list])
    y_vec = np.array([float(v) for v in inst["y"]])
    budget_spent = float(np.dot(c_cost_vec, y_vec))
    budget_spent_pct = 100.0 * budget_spent / B if B > 0 else 0.0

    # ------------------------------------------------------------------
    # Figure principale
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))

    # Axe graphe (70% de la largeur)
    ax_graph = fig.add_axes([0.0, 0.08, 0.68, 0.85])
    ax_graph.set_facecolor("#f8f9fa")

    nx.draw_networkx_edges(
        g, pos, ax=ax_graph,
        alpha=0.55,
        arrows=is_directed,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.6,
        edge_color="#555555",
        connectionstyle="arc3,rad=0.07",
    )
    nx.draw_networkx_nodes(
        g, pos, ax=ax_graph,
        node_color=colors,
        node_size=sizes,
        edgecolors="#2c3e50",
        linewidths=1.4,
    )

    # ID minimal centré sur chaque nœud
    nx.draw_networkx_labels(
        g, pos, ax=ax_graph,
        labels={nd: str(nd) for nd in g.nodes()},
        font_size=8, font_color="white", font_weight="bold",
    )

    # Labels détaillés positionnés À CÔTÉ des nœuds (avec flèche)
    all_xs = [v[0] for v in pos.values()]
    all_ys = [v[1] for v in pos.values()]
    cx = np.mean(all_xs)
    cy = np.mean(all_ys)
    x_range = (max(all_xs) - min(all_xs)) or 1.0
    y_range = (max(all_ys) - min(all_ys)) or 1.0
    offset_scale = max(x_range, y_range) * 0.38

    # Agrandir les limites pour que les annotations ne soient pas coupées
    margin = offset_scale * 1.3
    ax_graph.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
    ax_graph.set_ylim(min(all_ys) - margin, max(all_ys) + margin)

    for node in g.nodes():
        xc, yc = pos[node]
        label = node_label_multiline(node, inst)
        dx = xc - cx
        dy = yc - cy
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            dx_off, dy_off = offset_scale * 0.7, offset_scale * 0.7
        else:
            dx_off = (dx / dist) * offset_scale
            dy_off = (dy / dist) * offset_scale
        ax_graph.annotate(
            label,
            xy=(xc, yc),
            xytext=(xc + dx_off, yc + dy_off),
            fontsize=6.0,
            fontfamily="monospace",
            multialignment="left",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.92,
                      ec="#aaaaaa", lw=0.7),
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7,
                            shrinkA=0, shrinkB=6),
            zorder=5,
            annotation_clip=False,
        )

    # Légende couleurs
    low_col  = mcolors.to_hex(cm.YlOrRd(0.15))
    high_col = mcolors.to_hex(cm.YlOrRd(1.00))
    legend_elements = [
        mpatches.Patch(facecolor="#e74c3c", edgecolor="#2c3e50", label="Source (terminal)"),
        mpatches.Patch(facecolor="#e67e22", edgecolor="#2c3e50", label="Target (terminal)"),
        mpatches.Patch(facecolor=low_col,   edgecolor="#2c3e50", label="Réparable — budget π*·c faible"),
        mpatches.Patch(facecolor=high_col,  edgecolor="#2c3e50", label="Réparable — budget π*·c élevé"),
        mpatches.Patch(facecolor="#bdc3c7", edgecolor="#2c3e50", label="Non-réparable"),
    ]
    ax_graph.legend(handles=legend_elements, loc="lower left", fontsize=8, framealpha=0.9)
    ax_graph.set_title(
        f"Instance #{idx_in_dataset}  |  {inst.get('topology_type','?')}  |  "
        f"{'Dirigé' if is_directed else 'Non dirigé'}",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax_graph.axis("off")

    # Colorbar budget réparable (π*·c)
    _rep_nodes = inst.get("repairable_nodes", [])
    if any(nd in _rep_nodes for nd in g.nodes()):
        sm = cm.ScalarMappable(
            cmap=cm.YlOrRd,
            norm=mcolors.Normalize(vmin=0, vmax=max_budget_rep),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_graph, orientation="horizontal",
                            fraction=0.035, pad=0.06, aspect=35)
        cbar.set_label("Budget alloué  π* · c  par nœud réparable", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # ------------------------------------------------------------------
    # Panneau droit : métadonnées globales + tableau par nœud
    # ------------------------------------------------------------------
    ax_info = fig.add_axes([0.70, 0.08, 0.29, 0.85])
    ax_info.axis("off")

    terminals = inst.get("terminals", [])
    repairable = inst.get("repairable_nodes", [])

    meta_lines = [
        ("Topology", inst.get("topology_type", "?")),
        ("Instance #", str(idx_in_dataset)),
        ("", ""),
        ("── Paramètres globaux ──", ""),
        ("B  (budget)", f"{B:.4f}"),
        ("H  (horizon)", str(inst.get("H", "?"))),
        ("α  (ratio B/C_min)", f"{inst.get('alpha', float('nan')):.4f}"),
        ("C_min_path", f"{inst.get('C_min_path', float('nan')):.4f}"),
        ("C_total", f"{inst.get('C_total', float('nan')):.4f}"),
        ("Shortest path L", str(inst.get("shortest_path_length", "?"))),
        ("", ""),
        ("── Résultats solveur ──", ""),
        ("J*  (risque réel)", f"{inst.get('J_star', float('nan')):.6f}"),
        ("J_min (budget max)", f"{inst.get('J_min', float('nan')):.6f}"),
        ("J_max (budget=0)", f"{inst.get('J_max', float('nan')):.6f}"),
        ("", ""),
        ("── Budget effectif ──", ""),
        ("Σ(π*·c) dépensé", f"{budget_spent:.4f}"),
        ("% du budget", f"{budget_spent_pct:.1f}%"),
        ("", ""),
        ("── Structure ──", ""),
        ("n_nodes", str(inst.get("n_nodes", "?"))),
        ("n_edges", str(inst.get("n_edges", "?"))),
        ("Source", str(terminals[0] if terminals else "?")),
        ("Target", str(terminals[1] if len(terminals) > 1 else "?")),
        ("Réparables", str(repairable)),
        ("Tentatives", str(inst.get("attempts_needed", "?"))),
    ]

    y_cur = 0.99
    dy = 0.036
    for key, val in meta_lines:
        if key.startswith("──"):
            ax_info.text(0.02, y_cur, key, fontsize=8, fontweight="bold",
                         color="#2c3e50", va="top", transform=ax_info.transAxes)
        elif key == "":
            pass
        else:
            ax_info.text(0.02, y_cur, f"{key}:", fontsize=8, color="#555",
                         va="top", transform=ax_info.transAxes)
            ax_info.text(0.62, y_cur, val, fontsize=8, fontweight="bold",
                         color="#1a1a2e", va="top", transform=ax_info.transAxes)
        y_cur -= dy

    # Tableau des features par nœud
    y_cur -= 0.01
    ax_info.text(0.02, y_cur, "── Features par nœud (x) ──", fontsize=8,
                 fontweight="bold", color="#2c3e50", va="top", transform=ax_info.transAxes)
    y_cur -= dy

    headers = ["N", "p_fail", "c_cost", "deg⁻", "deg⁺", "d→T", "π*(y)"]
    col_xs = [0.0, 0.12, 0.27, 0.42, 0.55, 0.67, 0.80]
    for hx, h in zip(col_xs, headers):
        ax_info.text(hx, y_cur, h, fontsize=7, fontweight="bold",
                     color="#e74c3c" if h == "π*(y)" else "#2c3e50",
                     va="top", transform=ax_info.transAxes)
    y_cur -= dy * 0.8

    for node in sorted(g.nodes()):
        if y_cur < 0.0:
            break
        nd_idx = node_to_idx.get(node, 0)
        xv = inst["x"][nd_idx]
        yv = float(inst["y"][nd_idx])
        is_src = node == (terminals[0] if terminals else -1)
        is_tgt = node == (terminals[1] if len(terminals) > 1 else -1)
        row_color = "#c0392b" if is_src else ("#d35400" if is_tgt else "#1a1a2e")
        vals_str = [
            str(node),
            f"{xv[0]:.3f}",
            f"{xv[1]:.2f}",
            f"{int(xv[4])}",
            f"{int(xv[5])}",
            f"{xv[6]:.1f}",
            f"{yv:.4f}",
        ]
        for hx, v in zip(col_xs, vals_str):
            ax_info.text(hx, y_cur, v, fontsize=7,
                         color="#e74c3c" if hx == col_xs[-1] else row_color,
                         va="top", transform=ax_info.transAxes)
        y_cur -= dy * 0.85

    fig.suptitle(
        f"Inspection dataset  |  {Path(inst.get('topology_type','?')).name}  |  "
        f"J*={inst.get('J_star', 0):.4f}  |  B={B:.3f}  |  H={inst.get('H','?')}",
        fontsize=13, fontweight="bold", y=0.995,
    )

    plt.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure sauvegardée : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspecte une instance d'un dataset hybride")
    parser.add_argument("--dataset", default="datasetV6.json",
                        help="Chemin du dataset JSON")
    parser.add_argument("--index", type=int, default=None,
                        help="Index de l'instance à afficher (défaut: aléatoire)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed pour le tirage aléatoire (défaut: non reproductible)")
    parser.add_argument("--output", default=None,
                        help="Fichier image de sortie (défaut: inspect_<idx>.png)")
    args = parser.parse_args()

    instances = load_dataset(args.dataset)
    idx, inst = pick_instance(instances, index=args.index, seed=args.seed)

    out = args.output or f"inspect_instance_{idx}.png"
    print(f"Instance {idx}/{len(instances)-1} | topology={inst.get('topology_type')} "
          f"| n_nodes={inst.get('n_nodes')} | J*={inst.get('J_star'):.4f} | B={inst.get('B'):.4f}")
    draw_instance(inst, idx, out)


if __name__ == "__main__":
    main()
