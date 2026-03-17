import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def _load(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _safe_corr(x, y):
    if len(x) < 2:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    if np.isnan(c):
        return 0.0
    return float(c)


def _safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    if den <= 1e-12:
        return 0.0
    num = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - num / den)


def _bin_calibration(pred, mc, n_bins=8):
    pred = np.asarray(pred, dtype=float)
    mc = np.asarray(mc, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = []
    pred_mean = []
    mc_mean = []
    counts = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (pred >= lo) & (pred <= hi)
        else:
            mask = (pred >= lo) & (pred < hi)
        if np.any(mask):
            centers.append(0.5 * (lo + hi))
            pred_mean.append(float(np.mean(pred[mask])))
            mc_mean.append(float(np.mean(mc[mask])))
            counts.append(int(np.sum(mask)))
    return np.array(centers), np.array(pred_mean), np.array(mc_mean), np.array(counts)


def main():
    parser = argparse.ArgumentParser(
        description="Visualisation de la généralisation du modèle J (pred_j vs mc_j)"
    )
    parser.add_argument("--results-json", type=str,
                        default="benchmark_j_complex_gine.json")
    parser.add_argument("--out", type=str,
                        default="visu_j_gen_gine.png")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    data = _load(args.results_json)
    rows = data.get("rows", [])
    if not rows:
        raise ValueError("Aucune ligne disponible dans le fichier JSON")

    pred = np.array([row["pred_j"] for row in rows], dtype=float)
    mc = np.array([row["mc_j"] for row in rows], dtype=float)
    abs_err = np.array([row["abs_error"] for row in rows], dtype=float)
    signed_err = pred - mc
    n_nodes = np.array([row["n_nodes"] for row in rows], dtype=float)
    n_edges = np.array([row["n_edges"] for row in rows], dtype=float)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((pred - mc) ** 2)))
    corr = _safe_corr(pred, mc)
    r2 = _safe_r2(mc, pred)
    bias = float(np.mean(signed_err))

    top_bad_idx = np.argsort(-abs_err)[:5]

    fig = plt.figure(figsize=(16.5, 9.4))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.28)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_residual = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_calib = fig.add_subplot(gs[1, 0])
    ax_size = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[1, 2])

    # 1) Pred vs MC
    scatter = ax_scatter.scatter(
        mc,
        pred,
        c=n_nodes,
        cmap="viridis",
        s=35 + 1.2 * (n_edges - np.min(n_edges)),
        alpha=0.88,
        edgecolors="#263238",
        linewidths=0.45,
    )
    ax_scatter.plot([0, 1], [0, 1], "--", color="#D32F2F", linewidth=1.1)
    ax_scatter.set_xlim(0, 1)
    ax_scatter.set_ylim(0, 1)
    ax_scatter.set_xlabel("J Monte-Carlo (vérité)")
    ax_scatter.set_ylabel("J prédit")
    ax_scatter.set_title("Calibration globale")
    ax_scatter.grid(alpha=0.22)
    cbar = fig.colorbar(scatter, ax=ax_scatter, fraction=0.047, pad=0.03)
    cbar.set_label("Nombre de nœuds")

    # 2) Résidu vs vérité
    ax_residual.scatter(
        mc,
        signed_err,
        c=n_nodes,
        cmap="plasma",
        s=36,
        alpha=0.86,
        edgecolors="#263238",
        linewidths=0.4,
    )
    ax_residual.axhline(0.0, color="#263238", linestyle="--", linewidth=1.0)
    ax_residual.set_xlabel("J Monte-Carlo")
    ax_residual.set_ylabel("Erreur signée (pred - mc)")
    ax_residual.set_title("Biais selon la difficulté")
    ax_residual.grid(alpha=0.22)

    # 3) Distribution des erreurs
    ax_hist.hist(abs_err, bins=14, color="#FB8C00", edgecolor="white", alpha=0.9)
    ax_hist.axvline(mae, color="#263238", linestyle="-", linewidth=1.2, label=f"MAE={mae:.3f}")
    q90 = float(np.quantile(abs_err, 0.90))
    ax_hist.axvline(q90, color="#D32F2F", linestyle="--", linewidth=1.2, label=f"Q90={q90:.3f}")
    ax_hist.set_xlabel("|pred - mc|")
    ax_hist.set_ylabel("Nombre d'instances")
    ax_hist.set_title("Queue d'erreur")
    ax_hist.grid(alpha=0.22, axis="y")
    ax_hist.legend(fontsize=8)

    # 4) Courbe de calibration par bins
    _, pred_m, mc_m, counts = _bin_calibration(pred, mc, n_bins=8)
    if len(pred_m) > 0:
        ax_calib.plot([0, 1], [0, 1], "--", color="#D32F2F", linewidth=1.0, label="idéal")
        ax_calib.plot(pred_m, mc_m, "o-", color="#1E88E5", linewidth=1.6, label="bins")
        for x, y, n in zip(pred_m, mc_m, counts):
            ax_calib.text(x, y + 0.02, f"n={n}", fontsize=7, ha="center")
    ax_calib.set_xlim(0, 1)
    ax_calib.set_ylim(0, 1)
    ax_calib.set_xlabel("J prédit moyen (bin)")
    ax_calib.set_ylabel("J MC moyen (bin)")
    ax_calib.set_title("Calibration par paquets")
    ax_calib.grid(alpha=0.22)
    ax_calib.legend(fontsize=8)

    # 5) Erreur vs taille
    ax_size.scatter(
        n_nodes,
        abs_err,
        c=mc,
        cmap="cividis",
        s=42,
        alpha=0.9,
        edgecolors="#263238",
        linewidths=0.45,
    )
    z = np.polyfit(n_nodes, abs_err, deg=1)
    x_line = np.linspace(np.min(n_nodes), np.max(n_nodes), 100)
    y_line = z[0] * x_line + z[1]
    ax_size.plot(x_line, y_line, color="#D32F2F", linewidth=1.2)
    ax_size.set_xlabel("Nombre de nœuds")
    ax_size.set_ylabel("Erreur absolue")
    ax_size.set_title("Stabilité avec la taille")
    ax_size.grid(alpha=0.22)

    # 6) Panneau texte
    ax_text.axis("off")
    lines = [
        "Résumé généralisation J",
        "",
        f"Instances           : {len(rows)}",
        f"MAE                 : {mae:.4f}",
        f"RMSE                : {rmse:.4f}",
        f"Corr(Pred, MC)      : {corr:.4f}",
        f"R² (MC <- Pred)     : {r2:.4f}",
        f"Biais moyen         : {bias:+.4f}",
        "",
        "5 pires cas (|erreur|):",
    ]
    for rank, idx in enumerate(top_bad_idx, start=1):
        row = rows[int(idx)]
        lines.append(
            f"{rank}. id={row['id']}, n={row['n_nodes']}, "
            f"pred={row['pred_j']:.3f}, mc={row['mc_j']:.3f}, err={row['abs_error']:.3f}"
        )
    ax_text.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

    legend_items = [
        Line2D([0], [0], color="#D32F2F", linestyle="--", label="référence idéale"),
        Line2D([0], [0], color="#263238", linestyle="-", label="statistique observée"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

    fig.suptitle(
        "Diagnostic de généralisation du modèle J",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")

    print(f"Figure sauvegardée : {args.out}")
    print(f"MAE={mae:.4f} | RMSE={rmse:.4f} | Corr={corr:.4f} | R2={r2:.4f} | Bias={bias:+.4f}")


if __name__ == "__main__":
    main()