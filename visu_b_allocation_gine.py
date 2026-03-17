"""
Visualisation comparative : répartition du budget GINE_B vs solveur exact.

Pour une instance de chaque famille (ER, SP, Mesh), affiche côte à côte :
  - Graphe coloré par allocation exacte (solveur)
  - Graphe coloré par allocation GINE_B
  - Histogramme par nœud : exact vs GINE_B, avec budget dépensé

Usage :
    python visualize_gine_b_allocation.py --checkpoint checkpoints/gine_b_repartition_v7.pt
"""

import argparse
import copy
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import networkx as nx
import numpy as np
import torch
from torch.nn import Linear, ReLU, Sequential, Sigmoid
from torch_geometric.nn import GINEConv, global_add_pool
from monte_carlo_validation import simulate_monte_carlo


# ---------------------------------------------------------------------------
# Architecture GINE_B (identique à GINE_B_repartition.py)
# ---------------------------------------------------------------------------
class GINE_Allocation_Predictor(torch.nn.Module):
    def __init__(self, num_node_features=9, hidden_dim=64, edge_dim=1):
        super().__init__()
        self.conv1 = GINEConv(
            nn=Sequential(Linear(num_node_features, hidden_dim), ReLU(),
                          Linear(hidden_dim, hidden_dim), ReLU()),
            edge_dim=edge_dim,
        )
        self.conv2 = GINEConv(
            nn=Sequential(Linear(hidden_dim, hidden_dim), ReLU(),
                          Linear(hidden_dim, hidden_dim), ReLU()),
            edge_dim=edge_dim,
        )
        self.mlp_readout = Sequential(
            Linear(hidden_dim, hidden_dim // 2), ReLU(),
            Linear(hidden_dim // 2, 1), Sigmoid(),
        )

    def forward(self, x, edge_index, edge_attr, batch,
                B_total, terminal_mask=None, c_cost=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        pi_raw = self.mlp_readout(x).squeeze(-1)
        if terminal_mask is not None:
            pi_raw = pi_raw * (~terminal_mask.bool()).float()
        if c_cost is None:
            c_cost = torch.ones_like(pi_raw)
        node_exp = pi_raw * c_cost
        total_exp = global_add_pool(node_exp.view(-1, 1), batch).view(-1)
        ratio = torch.clamp(B_total[batch] / (total_exp[batch] + 1e-12), max=1.0)
        return pi_raw * ratio


def _load_model(path, device):
    model = GINE_Allocation_Predictor().to(device)
    raw = torch.load(path, map_location=device)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    remapped = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(remapped)
    model.eval()
    return model


def _infer(model, inst, device):
    """Retourne un tableau numpy (n_nodes,) = allocation prédite par GINE_B."""
    nodes = inst["graph"]["nodes"]
    edges = inst["graph"]["edges"]
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    x = torch.tensor(inst["x"], dtype=torch.float32)
    x[:, 1] /= 10.0;  x[:, 4] /= 15.0;  x[:, 5] /= 15.0
    x[:, 6] /= 15.0;  x[:, 7] /= 65.0;  x[:, 8] /= 25.0

    mapped = [[node_to_idx[u], node_to_idx[v]] for u, v in edges
              if u in node_to_idx and v in node_to_idx]
    edge_index = (torch.tensor(mapped, dtype=torch.long).t().contiguous()
                  if mapped else torch.empty((2, 0), dtype=torch.long))
    edge_attr = torch.ones((edge_index.size(1), 1))
    batch = torch.zeros(x.size(0), dtype=torch.long)
    B_total = torch.tensor([inst["B"]])

    tmask = torch.zeros(x.size(0), dtype=torch.bool)
    for t in inst["terminals"]:
        if t in node_to_idx:
            tmask[node_to_idx[t]] = True

    c_cost = torch.tensor([row[1] for row in inst["x"]])

    x, edge_index, edge_attr = x.to(device), edge_index.to(device), edge_attr.to(device)
    batch, B_total, tmask, c_cost = (batch.to(device), B_total.to(device),
                                     tmask.to(device), c_cost.to(device))

    with torch.no_grad():
        alloc = model(x, edge_index, edge_attr, batch, B_total, tmask, c_cost)
    return alloc.cpu().numpy()


def _graph_layout(inst):
    """
    Choisit un layout adapté :
    - mesh  → positions sur grille 2D extraites depuis topology_type
    - sp    → layout hiérarchique (dot-like via spring avec poids sur dist_to_target)
    - er    → spring layout
    """
    topo = inst.get("topology_type", "")
    nodes_list = inst["graph"]["nodes"]
    edges = inst["graph"]["edges"]
    G = nx.DiGraph()
    G.add_nodes_from(nodes_list)
    G.add_edges_from(edges)

    if topo.startswith("mesh_"):
        # Reconstruire la grille depuis le nom (ex: "mesh_3x4")
        try:
            suffix = topo.split("mesh_")[1]  # "3x4"
            m, n = [int(v) for v in suffix.split("x")]
            pos = {}
            for i, node in enumerate(sorted(nodes_list)):
                row, col = divmod(i, n)
                pos[node] = (col, -row)   # x=col, y=-row pour lisibilité top→bottom
            return G, pos
        except Exception:
            pass

    # SP : layout "layered" approximé via distance au target
    if topo.startswith("sp_"):
        node_to_idx = {n: i for i, n in enumerate(inst["graph"]["nodes"])}
        dist = {n: inst["x"][node_to_idx[n]][6] for n in nodes_list}
        max_dist = max(dist.values()) or 1.0
        # Grouper par distance, étaler horizontalement dans chaque couche
        layers = {}
        for n in nodes_list:
            d = int(dist[n]) if dist[n] < 900 else int(max_dist) + 1
            layers.setdefault(d, []).append(n)
        pos = {}
        for depth, layer_nodes in layers.items():
            for j, n in enumerate(sorted(layer_nodes)):
                pos[n] = (j - len(layer_nodes) / 2, -depth)
        return G, pos

    # ER et autres : spring
    pos = nx.spring_layout(G, seed=42, k=1.5 / max(1, np.sqrt(len(nodes_list))))
    return G, pos


def _draw_graph_alloc(ax, inst, alloc_values, title, cmap, norm,
                      node_size_base=600):
    """Dessine le graphe avec les nœuds colorés par alloc_values."""
    nodes_list = inst["graph"]["nodes"]
    terminals = set(inst["terminals"])
    source, target = inst["terminals"][0], inst["terminals"][1]
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}

    G, pos = _graph_layout(inst)

    # Couleurs des nœuds
    node_colors = []
    node_shapes_rep = [n for n in nodes_list if n not in terminals]
    for n in nodes_list:
        if n == source:
            node_colors.append("#2196F3")   # bleu = source
        elif n == target:
            node_colors.append("#4CAF50")   # vert = cible
        else:
            v = float(alloc_values[node_to_idx[n]])
            node_colors.append(cmap(norm(v)))

    # Taille proportionnelle à pi (nœuds réparables plus grands si pi élevé)
    sizes = []
    for n in nodes_list:
        if n in terminals:
            sizes.append(node_size_base * 1.3)
        else:
            v = float(alloc_values[node_to_idx[n]])
            sizes.append(node_size_base * (0.55 + 0.9 * v))

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, arrows=True,
                           arrowsize=12, edge_color="#888888",
                           connectionstyle="arc3,rad=0.08",
                           min_source_margin=12, min_target_margin=12)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=sizes, edgecolors="#333333", linewidths=0.8)

    # Labels : valeur de pi sur les nœuds réparables, symbole pour terminaux
    labels = {}
    for n in nodes_list:
        if n == source:
            labels[n] = "S"
        elif n == target:
            labels[n] = "T"
        else:
            labels[n] = f"{alloc_values[node_to_idx[n]]:.2f}"
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=6.5, font_color="white",
                            font_weight="bold")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.axis("off")


def _draw_barchart(ax, inst, exact_alloc, gine_alloc, family_label,
                   j_exact_mc=None, j_gine_mc=None):
    """
    Histogramme groupé par nœud réparable : exact (bleu) vs GINE_B (orange).
    Affiche aussi le budget dépensé par chaque méthode et J*.
    """
    nodes_list = inst["graph"]["nodes"]
    terminals = set(inst["terminals"])
    rep_nodes = [n for n in nodes_list if n not in terminals]
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}

    exact_pi = [exact_alloc[node_to_idx[n]] for n in rep_nodes]
    gine_pi  = [gine_alloc[node_to_idx[n]]  for n in rep_nodes]
    c_costs  = [inst["x"][node_to_idx[n]][1] for n in rep_nodes]

    B = inst["B"]
    spend_exact = sum(p * c for p, c in zip(exact_pi, c_costs))
    spend_gine  = sum(p * c for p, c in zip(gine_pi, c_costs))
    mae_val     = float(np.mean([abs(e - g) for e, g in zip(exact_pi, gine_pi)]))

    x_pos = np.arange(len(rep_nodes))
    width = 0.36

    bars_exact = ax.bar(x_pos - width / 2, exact_pi, width,
                        label=f"Exact  (Σpi·c={spend_exact:.2f})",
                        color="#2196F3", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_gine  = ax.bar(x_pos + width / 2, gine_pi, width,
                        label=f"GINE_B (Σpi·c={spend_gine:.2f})",
                        color="#FF9800", alpha=0.85, edgecolor="white", linewidth=0.5)

    # Ligne budget B
    ax.axhline(B / max(sum(c_costs), 1e-6), color="#E53935", linewidth=1.0,
               linestyle="--", label=f"B = {B:.2f}", alpha=0.7)

    # Labels sur les barres (valeur pi)
    for bar in list(bars_exact) + list(bars_gine):
        h = bar.get_height()
        if h > 0.04:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=5.5, color="#333333")

    # Nom des nœuds réparables en x
    short_labels = [f"n{n}" for n in rep_nodes]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, fontsize=6.5, rotation=45 if len(rep_nodes) > 8 else 0)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Prob. réparation π", fontsize=8)

    # Titre enrichi avec ΔJ_MC si disponible
    title_extra = ""
    if j_exact_mc is not None and j_gine_mc is not None:
        delta_j = j_gine_mc - j_exact_mc
        sign = "+" if delta_j >= 0 else ""
        quality = "↗ GINE_B pire" if delta_j > 1e-4 else ("↘ GINE_B mieux" if delta_j < -1e-4 else "≈ égal")
        title_extra = f"  |  ΔJ_MC={sign}{delta_j:.4f} ({quality})"
    ax.set_title(
        f"{family_label} — Exact vs GINE_B  |  MAE={mae_val:.4f}  |  J*={inst['J_star']:.4f}"
        + title_extra,
        fontsize=9, fontweight="bold"
    )
    ax.legend(fontsize=7, loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Annotation budget + MC
    mc_line = ""
    if j_exact_mc is not None and j_gine_mc is not None:
        delta_j = j_gine_mc - j_exact_mc
        sign = "+" if delta_j >= 0 else ""
        mc_line = (f"\n—\nJ_MC exact  : {j_exact_mc:.4f}"
                    f"\nJ_MC GINE_B : {j_gine_mc:.4f}"
                    f"\nΔJ_MC       : {sign}{delta_j:.4f}")
    ax.annotate(
        f"Budget B={B:.2f}\nExact: {spend_exact:.2f} ({100*spend_exact/B:.0f}%)\n"
        f"GINE_B: {spend_gine:.2f} ({100*spend_gine/B:.0f}%)"
        + mc_line,
        xy=(0.01, 0.97), xycoords="axes fraction",
        va="top", ha="left", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="#cccccc"),
    )


def _pick_instance(dataset_path, preferred_idx=None, min_nodes=8):
    """Charge et retourne une instance avec suffisamment de nœuds."""
    with open(dataset_path) as f:
        data = json.load(f)
    instances = data["instances"]

    if preferred_idx is not None and 0 <= preferred_idx < len(instances):
        return instances[preferred_idx]

    # Choisir la plus grande instance disponible avec un J* intéressant (>0.02)
    candidates = [inst for inst in instances
                  if inst.get("n_nodes", 0) >= min_nodes
                  and inst.get("J_star", 0.0) > 0.02]
    if not candidates:
        candidates = [inst for inst in instances if inst.get("n_nodes", 0) >= min_nodes]
    if not candidates:
        return instances[0]
    return sorted(candidates, key=lambda i: (-i.get("n_nodes", 0),
                                              abs(i.get("J_star", 0) - 0.25)))[0]


def _simulate_mc_for_alloc(inst, alloc_array, n_sims, seed=42):
    """
    Évalue J (downtime moyen / H) par Monte-Carlo pour une allocation donnée.
    Remplace inst["y"] par alloc_array (indexé sur inst["graph"]["nodes"]).
    """
    inst_copy = dict(inst)                            # shallow copy, n'écrase pas l'original
    y_full = [float(alloc_array[i]) for i in range(len(inst["graph"]["nodes"]))]
    inst_copy["y"] = y_full
    np.random.seed(seed)
    return simulate_monte_carlo(inst_copy, n_sims=n_sims)



def main():
    parser = argparse.ArgumentParser(
        description="Visualisation allocation GINE_B vs solveur exact, une instance par famille"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/gine_b_repartition_v7.pt",
                        help="Checkpoint GINE_B (.pt)")
    parser.add_argument("--er-json",   type=str, default="datasetV7_er.json")
    parser.add_argument("--sp-json",   type=str, default="datasetV7_sp.json")
    parser.add_argument("--mesh-json", type=str, default="datasetV7_mesh.json")
    parser.add_argument("--idx-er",    type=int, default=None,
                        help="Index instance ER (auto si absent)")
    parser.add_argument("--idx-sp",    type=int, default=None,
                        help="Index instance SP")
    parser.add_argument("--idx-mesh",  type=int, default=None,
                        help="Index instance Mesh")
    parser.add_argument("--out", type=str,
                        default="visu_b_allocation_gine.png",
                        help="Fichier image de sortie")
    parser.add_argument("--dpi", type=int, default=180,
                        help="Résolution de l'image (défaut 180)")
    parser.add_argument("--mc-sims", type=int, default=2000,
                        help="Nombre de simulations Monte-Carlo pour ΔJ (défaut 2000)")
    parser.add_argument("--no-mc", action="store_true",
                        help="Désactiver le calcul Monte-Carlo de ΔJ")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = _load_model(args.checkpoint, device)

    families = [
        ("ER  (Erdős-Rényi)",      args.er_json,   args.idx_er),
        ("SP  (Série-Parallèle)",  args.sp_json,   args.idx_sp),
        ("Mesh (Grille)",          args.mesh_json, args.idx_mesh),
    ]

    instances, preds, labels = [], [], []
    for label, json_path, idx in families:
        inst = _pick_instance(json_path, preferred_idx=idx)
        alloc_pred = _infer(model, inst, device)
        instances.append(inst)
        preds.append(alloc_pred)
        labels.append(label)
        topo = inst.get("topology_type", "?")
        n = inst.get("n_nodes", "?")
        j = inst.get("J_star", 0.0)
        print(f"  [{label.split()[0]}] {topo} | {n} nœuds | J*={j:.4f} | B={inst['B']:.2f}")

    # Calcul Monte-Carlo des J pour exact et GINE_B
    mc_results = []   # liste de (j_exact_mc, j_gine_mc) ou (None, None)
    if not args.no_mc:
        print(f"\nSimulation Monte-Carlo ({args.mc_sims} sims / instance)...")
        for i, (inst, pred, label) in enumerate(zip(instances, preds, labels)):
            exact_alloc = np.array(inst["y"], dtype=float)
            j_exact_mc = _simulate_mc_for_alloc(inst, exact_alloc, args.mc_sims, seed=42 + i)
            j_gine_mc  = _simulate_mc_for_alloc(inst, pred,        args.mc_sims, seed=42 + i)
            mc_results.append((j_exact_mc, j_gine_mc))
            delta_j = j_gine_mc - j_exact_mc
            sign = "+" if delta_j >= 0 else ""
            print(f"  [{label.split()[0]}] J_exact_MC={j_exact_mc:.4f}  "
                  f"J_GINE_MC={j_gine_mc:.4f}  ΔJ={sign}{delta_j:.4f}")
    else:
        mc_results = [(None, None)] * len(instances)

    # -------------------------------------------------------------------------
    # Figure : 3 lignes × 3 colonnes
    #   col 0 : graphe allocation exacte
    #   col 1 : graphe allocation GINE_B
    #   col 2 : barchart comparatif
    # -------------------------------------------------------------------------
    cmap = plt.get_cmap("YlOrRd")
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(18, 15))
    fig.patch.set_facecolor("#F8F9FA")

    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        width_ratios=[1, 1, 1.5],
        hspace=0.40,
        wspace=0.30,
        left=0.04, right=0.97, top=0.93, bottom=0.05,
    )

    for row, (inst, pred, label) in enumerate(zip(instances, preds, labels)):
        nodes_list = inst["graph"]["nodes"]
        node_to_idx = {n: i for i, n in enumerate(nodes_list)}
        exact_alloc = np.array(inst["y"], dtype=float)
        j_exact_mc, j_gine_mc = mc_results[row]

        ax_exact = fig.add_subplot(gs[row, 0])
        ax_gine  = fig.add_subplot(gs[row, 1])
        ax_bar   = fig.add_subplot(gs[row, 2])

        _draw_graph_alloc(ax_exact, inst, exact_alloc,
                          f"{label}\nSolveur exact", cmap, norm)
        _draw_graph_alloc(ax_gine, inst, pred,
                          f"{label}\nGINE_B", cmap, norm)
        _draw_barchart(ax_bar, inst, exact_alloc, pred, label,
                       j_exact_mc=j_exact_mc, j_gine_mc=j_gine_mc)

    # Colorbar commune pour les graphes
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.02, 0.04, 0.01, 0.88])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("π (prob. de réparation)", fontsize=9, rotation=90, labelpad=8)
    cbar.ax.tick_params(labelsize=7)

    # Légende terminaux (commune)
    legend_handles = [
        mpatches.Patch(color="#2196F3", label="Source (S)"),
        mpatches.Patch(color="#4CAF50", label="Cible (T)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.40, 0.005))

    fig.suptitle(
        "Comparaison répartition budget : Solveur exact vs GINE_B\n"
        f"Checkpoint : {args.checkpoint}",
        fontsize=13, fontweight="bold", y=0.97
    )

    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nFigure sauvegardée : {args.out}")

    # -------------------------------------------------------------------------
    # Résumé chiffré dans le terminal
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'Famille':<24} {'n_nodes':>7} {'J*':>8} "
          f"{'MAE(π)':>9} {'ΔBudget%':>10} {'J_exact_MC':>12} {'J_GINE_MC':>11} {'ΔJ_MC':>9}")
    print("-" * 72)
    for (inst, pred, label), (j_exact_mc, j_gine_mc) in zip(
            zip(instances, preds, labels), mc_results):
        nodes_list = inst["graph"]["nodes"]
        terminals = set(inst["terminals"])
        node_to_idx = {n: i for i, n in enumerate(nodes_list)}
        rep_nodes = [n for n in nodes_list if n not in terminals]
        exact_alloc = np.array(inst["y"], dtype=float)
        c_costs = [inst["x"][node_to_idx[n]][1] for n in rep_nodes]
        exact_pi = [exact_alloc[node_to_idx[n]] for n in rep_nodes]
        gine_pi  = [pred[node_to_idx[n]] for n in rep_nodes]
        mae = float(np.mean([abs(e - g) for e, g in zip(exact_pi, gine_pi)]))
        spend_e = sum(p * c for p, c in zip(exact_pi, c_costs))
        spend_g = sum(p * c for p, c in zip(gine_pi, c_costs))
        delta_b = 100.0 * abs(spend_e - spend_g) / max(inst["B"], 1e-6)

        if j_exact_mc is not None and j_gine_mc is not None:
            delta_j = j_gine_mc - j_exact_mc
            sign = "+" if delta_j >= 0 else ""
            mc_str = f"{j_exact_mc:>12.4f} {j_gine_mc:>11.4f} {sign}{delta_j:>8.4f}"
        else:
            mc_str = f"{'—':>12} {'—':>11} {'—':>9}"

        print(f"{label.split('(')[0].strip():<24} {inst['n_nodes']:>7} "
              f"{inst['J_star']:>8.4f} {mae:>9.4f} {delta_b:>9.1f}%  {mc_str}")
    print("=" * 72)


if __name__ == "__main__":
    main()
