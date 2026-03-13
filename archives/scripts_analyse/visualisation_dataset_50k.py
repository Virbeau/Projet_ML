#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np
import matplotlib.pyplot as plt


def family_from_topology(topology_type):
    if topology_type.startswith("mesh"):
        return "mesh"
    if topology_type.startswith("sp"):
        return "sp"
    if topology_type.startswith("er"):
        return "er"
    return "other"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_dataset_cache(instances):
    B = np.array([float(i["B"]) for i in instances], dtype=float)
    H = np.array([int(i["H"]) for i in instances], dtype=int)
    N = np.array([int(i["n_nodes"]) for i in instances], dtype=int)
    E = np.array([int(i["n_edges"]) for i in instances], dtype=int)
    J = np.array([float(i["J_star"]) for i in instances], dtype=float)
    T = [i["topology_type"] for i in instances]
    F = np.array([family_from_topology(t) for t in T])

    densities = []
    for i in instances:
        n = int(i["n_nodes"])
        e = int(i["n_edges"])
        is_dir = bool(i.get("graph", {}).get("is_directed", True))
        if n <= 1:
            densities.append(0.0)
            continue
        if is_dir:
            densities.append(e / (n * (n - 1)))
        else:
            densities.append(e / (n * (n - 1) / 2))

    D = np.array(densities, dtype=float)
    return B, H, N, E, J, T, F, D


def plot_hist_and_cdf(J, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    axes[0].hist(J, bins=70, color="#1f77b4", alpha=0.85, edgecolor="white")
    axes[0].set_title("Distribution de J* (50k)")
    axes[0].set_xlabel("J* (probabilité d'échec)")
    axes[0].set_ylabel("Nombre d'instances")
    axes[0].grid(alpha=0.2)

    x = np.sort(J)
    y = np.arange(1, len(x) + 1) / len(x)
    axes[1].plot(x, y, color="#d62728", linewidth=2)
    axes[1].set_title("CDF empirique de J*")
    axes[1].set_xlabel("J*")
    axes[1].set_ylabel("P(J* ≤ x)")
    axes[1].grid(alpha=0.25)

    save_fig(fig, out_dir / "01_distribution_J_hist_cdf.png")


def plot_risk_bands(J, out_dir: Path):
    bands = {
        "[0,0.1]": np.mean(J <= 0.1),
        "(0.1,0.3]": np.mean((J > 0.1) & (J <= 0.3)),
        "(0.3,0.6]": np.mean((J > 0.3) & (J <= 0.6)),
        "(0.6,0.9]": np.mean((J > 0.6) & (J <= 0.9)),
        "(0.9,1.0]": np.mean(J > 0.9),
    }

    labels = list(bands.keys())
    values = [bands[k] * 100 for k in labels]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar(labels, values, color=["#4daf4a", "#a6d854", "#ffd92f", "#fc8d62", "#e41a1c"])
    ax.set_title("Bandes de risque de défaillance (J*)")
    ax.set_ylabel("Part du dataset (%)")
    ax.set_xlabel("Intervalle de J*")
    ax.grid(axis="y", alpha=0.25)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    save_fig(fig, out_dir / "02_risk_bands_bar.png")


def plot_family_boxplot(J, F, out_dir: Path):
    families = ["mesh", "sp", "er"]
    data = [J[F == fam] for fam in families]

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    bp = ax.boxplot(data, tick_labels=families, patch_artist=True, showfliers=False)
    colors = ["#66c2a5", "#fc8d62", "#8da0cb"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)

    ax.set_title("J* par famille de topologie")
    ax.set_ylabel("J*")
    ax.set_xlabel("Famille")
    ax.grid(axis="y", alpha=0.25)

    means = [np.mean(d) for d in data]
    for i, m in enumerate(means, start=1):
        ax.text(i, min(0.98, m + 0.03), f"μ={m:.3f}", ha="center", fontsize=9)

    save_fig(fig, out_dir / "03_family_boxplot.png")


def plot_heatmap_H_N(H, N, J, out_dir: Path):
    h_vals = np.sort(np.unique(H))
    n_vals = np.sort(np.unique(N))

    mean_grid = np.full((len(h_vals), len(n_vals)), np.nan)
    count_grid = np.zeros((len(h_vals), len(n_vals)), dtype=int)

    for i, h in enumerate(h_vals):
        for j, n in enumerate(n_vals):
            mask = (H == h) & (N == n)
            c = int(np.sum(mask))
            count_grid[i, j] = c
            if c > 0:
                mean_grid[i, j] = np.mean(J[mask])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    im1 = axes[0].imshow(mean_grid, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Heatmap moyenne J* (H x n_nodes)")
    axes[0].set_xlabel("n_nodes")
    axes[0].set_ylabel("H")
    axes[0].set_xticks(range(len(n_vals)))
    axes[0].set_xticklabels(n_vals)
    axes[0].set_yticks(range(len(h_vals)))
    axes[0].set_yticklabels(h_vals)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(count_grid, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Heatmap volume d'instances (H x n_nodes)")
    axes[1].set_xlabel("n_nodes")
    axes[1].set_ylabel("H")
    axes[1].set_xticks(range(len(n_vals)))
    axes[1].set_xticklabels(n_vals)
    axes[1].set_yticks(range(len(h_vals)))
    axes[1].set_yticklabels(h_vals)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    save_fig(fig, out_dir / "04_heatmap_H_vs_nnodes.png")


def plot_heatmap_B_H(B, H, J, out_dir: Path):
    b_bins = [1.0, 2.0, 3.0, 4.0, 5.0]
    b_labels = ["[1,2)", "[2,3)", "[3,4)", "[4,5]"]
    h_vals = np.sort(np.unique(H))

    mean_grid = np.full((len(h_vals), len(b_labels)), np.nan)
    for i, h in enumerate(h_vals):
        for j in range(len(b_labels)):
            low = b_bins[j]
            high = b_bins[j + 1]
            if j < len(b_labels) - 1:
                mask = (H == h) & (B >= low) & (B < high)
            else:
                mask = (H == h) & (B >= low) & (B <= high)
            if np.any(mask):
                mean_grid[i, j] = np.mean(J[mask])

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    im = ax.imshow(mean_grid, aspect="auto", origin="lower", cmap="plasma", vmin=0, vmax=1)
    ax.set_title("Moyenne J* par budget B et horizon H")
    ax.set_xlabel("Tranches de budget B")
    ax.set_ylabel("H")
    ax.set_xticks(range(len(b_labels)))
    ax.set_xticklabels(b_labels)
    ax.set_yticks(range(len(h_vals)))
    ax.set_yticklabels(h_vals)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_fig(fig, out_dir / "05_heatmap_B_vs_H.png")


def plot_topology_bars(T, J, out_dir: Path):
    groups = defaultdict(list)
    for t, j in zip(T, J):
        groups[t].append(j)

    rows = []
    for t, vals in groups.items():
        arr = np.array(vals)
        rows.append((t, len(vals), float(np.mean(arr)), float(np.std(arr))))

    rows_sorted = sorted(rows, key=lambda x: x[2], reverse=True)
    names = [r[0] for r in rows_sorted]
    means = [r[2] for r in rows_sorted]
    counts = [r[1] for r in rows_sorted]

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    axes[0].bar(range(len(names)), means, color="#377eb8")
    axes[0].set_title("Topologies classées par J* moyen (décroissant)")
    axes[0].set_ylabel("J* moyen")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(range(len(names)), counts, color="#4daf4a")
    axes[1].set_title("Nombre d'instances par topologie")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Topologie")
    axes[1].grid(axis="y", alpha=0.25)

    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=70, ha="right")

    save_fig(fig, out_dir / "06_topology_rank_and_counts.png")


def plot_budget_network_segments(B, N, H, J, out_dir: Path):
    b_q25 = float(np.percentile(B, 25))
    b_q75 = float(np.percentile(B, 75))
    n_q25 = float(np.percentile(N, 25))
    n_q75 = float(np.percentile(N, 75))
    h_q75 = float(np.percentile(H, 75))

    segments = {
        "small_budget + small_network": (B <= b_q25) & (N <= n_q25),
        "small_budget + large_network": (B <= b_q25) & (N >= n_q75),
        "large_budget + large_network": (B >= b_q75) & (N >= n_q75),
        "high_H + large_network": (H >= h_q75) & (N >= n_q75),
        "high_H + small_network": (H >= h_q75) & (N <= n_q25),
    }

    labels, means, p90, counts = [], [], [], []
    for k, mask in segments.items():
        if np.sum(mask) == 0:
            continue
        vals = J[mask]
        labels.append(k)
        means.append(float(np.mean(vals)))
        p90.append(float(np.percentile(vals, 90)))
        counts.append(int(np.sum(mask)))

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    w = 0.38
    ax.bar(x - w / 2, means, width=w, label="Moyenne J*", color="#984ea3")
    ax.bar(x + w / 2, p90, width=w, label="P90 J*", color="#ff7f00")
    ax.set_title("Segments clés: budget / taille réseau / horizon")
    ax.set_ylabel("J*")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for xi, c in zip(x, counts):
        ax.text(xi, 1.01, f"n={c}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 1.08)
    save_fig(fig, out_dir / "07_key_segments_budget_network_H.png")


def plot_scatter_budget_density(B, D, J, F, out_dir: Path):
    rng = np.random.default_rng(42)
    n = len(J)
    sample_size = min(12000, n)
    idx = rng.choice(n, size=sample_size, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    colors = {"mesh": "#1b9e77", "sp": "#d95f02", "er": "#7570b3", "other": "#666666"}

    for fam in ["mesh", "sp", "er"]:
        m = (F[idx] == fam)
        axes[0].scatter(B[idx][m], J[idx][m], s=9, alpha=0.28, label=fam, c=colors[fam], edgecolors="none")

    axes[0].set_title("Nuage B vs J* (échantillon 12k)")
    axes[0].set_xlabel("Budget B")
    axes[0].set_ylabel("J*")
    axes[0].grid(alpha=0.22)
    axes[0].legend(title="Famille")

    for fam in ["mesh", "sp", "er"]:
        m = (F[idx] == fam)
        axes[1].scatter(D[idx][m], J[idx][m], s=9, alpha=0.28, label=fam, c=colors[fam], edgecolors="none")

    axes[1].set_title("Nuage densité vs J* (échantillon 12k)")
    axes[1].set_xlabel("Densité du graphe")
    axes[1].set_ylabel("J*")
    axes[1].grid(alpha=0.22)
    axes[1].legend(title="Famille")

    save_fig(fig, out_dir / "08_scatter_budget_density_vs_J.png")


def plot_correlation_matrix(B, H, N, E, D, J, out_dir: Path):
    X = np.vstack([B, H, N, E, D, J]).T
    corr = np.corrcoef(X, rowvar=False)
    labels = ["B", "H", "n_nodes", "n_edges", "density", "J*"]

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Matrice de corrélation")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, out_dir / "09_correlation_matrix.png")


def main():
    parser = argparse.ArgumentParser(description="Génération de visuels analytiques pour un dataset hybride")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_hybrid_mesh_sp_er_v2_1000.json",
        help="Chemin vers le dataset JSON"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Dossier de sortie des figures (défaut: visuals_<nom_dataset>)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"analysis_visuals_{dataset_path.stem}")
    ensure_dir(out_dir)

    with dataset_path.open("r") as f:
        data = json.load(f)

    instances = data["instances"]
    B, H, N, E, J, T, F, D = build_dataset_cache(instances)

    plot_hist_and_cdf(J, out_dir)
    plot_risk_bands(J, out_dir)
    plot_family_boxplot(J, F, out_dir)
    plot_heatmap_H_N(H, N, J, out_dir)
    plot_heatmap_B_H(B, H, J, out_dir)
    plot_topology_bars(T, J, out_dir)
    plot_budget_network_segments(B, N, H, J, out_dir)
    plot_scatter_budget_density(B, D, J, F, out_dir)
    plot_correlation_matrix(B, H, N, E, D, J, out_dir)

    print("Visuels générés dans:", out_dir)
    for p in sorted(out_dir.glob("*.png")):
        print("-", p.name)


if __name__ == "__main__":
    main()
