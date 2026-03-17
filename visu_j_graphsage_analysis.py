"""
Analyse comparative GraphSAGE sur benchmarks easy-large et complex.
Produit une figure multi-panneaux montrant calibration, biais, erreurs
et comparaison des deux types de topologies côte à côte.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


# ─── helpers ────────────────────────────────────────────────────────────────

def _load(path):
    with open(path) as f:
        return json.load(f)


def _safe_corr(x, y):
    if len(x) < 2:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def _safe_r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    return 0.0 if den <= 1e-12 else float(1.0 - np.sum((y_true - y_pred) ** 2) / den)


def _bin_calibration(pred, mc, n_bins=8):
    pred, mc = np.asarray(pred, float), np.asarray(mc, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, pred_m, mc_m, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (pred >= lo) & (pred <= hi)
        if np.any(mask):
            centers.append(0.5 * (lo + hi))
            pred_m.append(float(np.mean(pred[mask])))
            mc_m.append(float(np.mean(mc[mask])))
            counts.append(int(np.sum(mask)))
    return np.array(centers), np.array(pred_m), np.array(mc_m), np.array(counts)


def _extract(rows):
    pred  = np.array([r["pred_j"]    for r in rows], float)
    mc    = np.array([r["mc_j"]      for r in rows], float)
    err   = np.array([r["abs_error"] for r in rows], float)
    nodes = np.array([r["n_nodes"]   for r in rows], float)
    edges = np.array([r["n_edges"]   for r in rows], float)
    return pred, mc, err, nodes, edges


def _metrics(pred, mc):
    mae  = float(np.mean(np.abs(pred - mc)))
    rmse = float(np.sqrt(np.mean((pred - mc) ** 2)))
    corr = _safe_corr(pred, mc)
    r2   = _safe_r2(mc, pred)
    bias = float(np.mean(pred - mc))
    return dict(mae=mae, rmse=rmse, corr=corr, r2=r2, bias=bias)


# ─── palette ────────────────────────────────────────────────────────────────
C_EASY    = "#1E88E5"   # bleu
C_COMPLEX = "#E53935"   # rouge
C_REF     = "#263238"   # gris foncé


# ─── figure ─────────────────────────────────────────────────────────────────

def build_figure(easy_rows, complex_rows, title, out_path, dpi):

    pred_e, mc_e, err_e, nodes_e, edges_e = _extract(easy_rows)
    pred_c, mc_c, err_c, nodes_c, edges_c = _extract(complex_rows)
    m_e = _metrics(pred_e, mc_e)
    m_c = _metrics(pred_c, mc_c)

    fig = plt.figure(figsize=(19, 13))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        hspace=0.38,
        wspace=0.30,
        top=0.95, bottom=0.07, left=0.06, right=0.97,
    )

    # ── ligne 0 : scatter pred vs MC ──────────────────────────────────────
    for col, (pred, mc, nodes, edges, m, color, label) in enumerate([
        (pred_e, mc_e, nodes_e, edges_e, m_e, C_EASY,    "Easy-large"),
        (pred_c, mc_c, nodes_c, edges_c, m_c, C_COMPLEX, "Complex"),
    ]):
        ax = fig.add_subplot(gs[0, col * 2: col * 2 + 2])
        sc = ax.scatter(mc, pred,
                        c=nodes, cmap="viridis",
                        s=30 + 1.0 * (edges - edges.min()),
                        alpha=0.85, edgecolors=C_REF, linewidths=0.4)
        lim = max(mc.max(), pred.max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color="#D32F2F", lw=1.1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("J Monte-Carlo (vérité)", fontsize=9)
        ax.set_ylabel("J prédit", fontsize=9)
        ax.set_title(f"Calibration — {label}", fontsize=10, fontweight="bold", color=color)
        ax.grid(alpha=0.20)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("Nœuds", fontsize=7)
        ax.text(0.03, 0.97,
                f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}\nCorr={m['corr']:.3f}  Biais={m['bias']:+.4f}",
                va="top", ha="left", fontsize=7.5,
                transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))

    # ── ligne 1 : distribution des erreurs (côte à côte) ─────────────────
    ax_hist = fig.add_subplot(gs[1, 0:2])
    bins = np.linspace(0, max(err_e.max(), err_c.max()) * 1.05, 20)
    ax_hist.hist(err_e, bins=bins, color=C_EASY,    alpha=0.72, edgecolor="white", label="Easy-large")
    ax_hist.hist(err_c, bins=bins, color=C_COMPLEX, alpha=0.72, edgecolor="white", label="Complex")
    ax_hist.axvline(m_e["mae"], color=C_EASY,    lw=1.4, linestyle="--",
                    label=f"MAE easy={m_e['mae']:.4f}")
    ax_hist.axvline(m_c["mae"], color=C_COMPLEX, lw=1.4, linestyle="--",
                    label=f"MAE complex={m_c['mae']:.4f}")
    ax_hist.set_xlabel("|pred − mc|", fontsize=9)
    ax_hist.set_ylabel("Nb instances", fontsize=9)
    ax_hist.set_title("Distribution des erreurs absolues", fontsize=10)
    ax_hist.grid(alpha=0.18, axis="y")
    ax_hist.legend(fontsize=7.5)

    # ── ligne 1 : erreur vs taille ────────────────────────────────────────
    ax_size = fig.add_subplot(gs[1, 2:4])
    for pred, mc, err, nodes, color, label, marker in [
        (pred_e, mc_e, err_e, nodes_e, C_EASY,    "Easy-large", "o"),
        (pred_c, mc_c, err_c, nodes_c, C_COMPLEX, "Complex",    "s"),
    ]:
        ax_size.scatter(nodes, err, c=color, s=38, alpha=0.82,
                        edgecolors=C_REF, linewidths=0.35, marker=marker, label=label)
        z = np.polyfit(nodes, err, 1)
        x_line = np.linspace(nodes.min(), nodes.max(), 100)
        ax_size.plot(x_line, z[0] * x_line + z[1], color=color, lw=1.2, linestyle="--")
    ax_size.set_xlabel("Nombre de nœuds", fontsize=9)
    ax_size.set_ylabel("Erreur absolue", fontsize=9)
    ax_size.set_title("Stabilité selon la taille du graphe", fontsize=10)
    ax_size.grid(alpha=0.18)
    ax_size.legend(fontsize=8)

    # ── ligne 2 : calibration par bins + résidu signé + panneau texte ────
    ax_calib = fig.add_subplot(gs[2, 0])
    ax_residual = fig.add_subplot(gs[2, 1])
    ax_text_e   = fig.add_subplot(gs[2, 2])
    ax_text_c   = fig.add_subplot(gs[2, 3])

    # calibration par bins (les deux topologies)
    ax_calib.plot([0, 1], [0, 1], "--", color="#D32F2F", lw=1.0, label="idéal", zorder=0)
    for pred, mc, color, label in [
        (pred_e, mc_e, C_EASY,    "Easy-large"),
        (pred_c, mc_c, C_COMPLEX, "Complex"),
    ]:
        _, pm, mm, cnt = _bin_calibration(pred, mc, n_bins=7)
        if len(pm):
            ax_calib.plot(pm, mm, "o-", color=color, lw=1.5, label=label)
            for x, y, n in zip(pm, mm, cnt):
                ax_calib.text(x, y + 0.012, f"{n}", fontsize=6.5,
                              ha="center", color=color)
    ax_calib.set_xlim(0, 1); ax_calib.set_ylim(0, 1)
    ax_calib.set_xlabel("J prédit moyen (bin)", fontsize=9)
    ax_calib.set_ylabel("J MC moyen (bin)", fontsize=9)
    ax_calib.set_title("Calibration par paquets", fontsize=10)
    ax_calib.grid(alpha=0.18)
    ax_calib.legend(fontsize=7.5)

    # résidu signé vs vérité (les deux topologies)
    ax_residual.axhline(0.0, color=C_REF, lw=1.0, linestyle="--")
    for pred, mc, nodes, color, label in [
        (pred_e, mc_e, nodes_e, C_EASY,    "Easy-large"),
        (pred_c, mc_c, nodes_c, C_COMPLEX, "Complex"),
    ]:
        ax_residual.scatter(mc, pred - mc, c=color, s=28, alpha=0.75,
                            edgecolors=C_REF, linewidths=0.3, label=label)
    ax_residual.set_xlabel("J Monte-Carlo", fontsize=9)
    ax_residual.set_ylabel("Erreur signée (pred − mc)", fontsize=9)
    ax_residual.set_title("Biais selon la difficulté", fontsize=10)
    ax_residual.grid(alpha=0.18)
    ax_residual.legend(fontsize=7.5)

    # panneau résumé par topologie
    for ax_txt, m, rows, label, color in [
        (ax_text_e, m_e, easy_rows,    "Easy-large", C_EASY),
        (ax_text_c, m_c, complex_rows, "Complex",    C_COMPLEX),
    ]:
        ax_txt.axis("off")
        top5 = sorted(rows, key=lambda r: r["abs_error"], reverse=True)[:5]
        lines = [
            f"── GraphSAGE  {label} ──",
            "",
            f"n instances   : {len(rows)}",
            f"MAE           : {m['mae']:.4f}",
            f"RMSE          : {m['rmse']:.4f}",
            f"Corr(pred,MC) : {m['corr']:.4f}",
            f"R²            : {m['r2']:.4f}",
            f"Biais moyen   : {m['bias']:+.4f}",
            "",
            "5 pires cas :",
        ]
        for rank, r in enumerate(top5, 1):
            lines.append(
                f"  {rank}. id={r['id']} n={r['n_nodes']} "
                f"pred={r['pred_j']:.3f} mc={r['mc_j']:.3f} err={r['abs_error']:.3f}"
            )
        ax_txt.text(0.02, 0.99, "\n".join(lines),
                    va="top", ha="left", fontsize=8, family="monospace",
                    transform=ax_txt.transAxes,
                    bbox=dict(facecolor="#F5F5F5", alpha=0.9,
                              edgecolor=color, linewidth=1.2, pad=4))

    # légende globale
    legend_items = [
        Patch(facecolor=C_EASY,    label="Easy-large", alpha=0.85),
        Patch(facecolor=C_COMPLEX, label="Complex",    alpha=0.85),
        Line2D([0], [0], color="#D32F2F", linestyle="--", label="Référence idéale"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.005))

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Figure sauvegardée : {out_path}")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse GraphSAGE — easy-large vs complex (pred_j vs mc_j)"
    )
    parser.add_argument("--easy-json",    type=str,
                        default="benchmark_j_easy_graphsage.json")
    parser.add_argument("--complex-json", type=str,
                        default="benchmark_j_complex_graphsage.json")
    parser.add_argument("--out",          type=str,
                        default="visu_j_graphsage_analysis.png")
    parser.add_argument("--dpi",          type=int, default=180)
    args = parser.parse_args()

    easy_rows    = _load(args.easy_json).get("rows", [])
    complex_rows = _load(args.complex_json).get("rows", [])

    if not easy_rows:
        raise ValueError(f"Aucune ligne dans {args.easy_json}")
    if not complex_rows:
        raise ValueError(f"Aucune ligne dans {args.complex_json}")

    print(f"Easy-large : {len(easy_rows)} instances")
    print(f"Complex    : {len(complex_rows)} instances")

    build_figure(
        easy_rows, complex_rows,
        title="Analyse GraphSAGE — Easy-large vs Complex (J*)",
        out_path=args.out,
        dpi=args.dpi,
    )

    # résumé terminal
    pred_e = np.array([r["pred_j"] for r in easy_rows])
    mc_e   = np.array([r["mc_j"]   for r in easy_rows])
    pred_c = np.array([r["pred_j"] for r in complex_rows])
    mc_c   = np.array([r["mc_j"]   for r in complex_rows])
    for label, pred, mc in [("Easy-large", pred_e, mc_e), ("Complex", pred_c, mc_c)]:
        m = _metrics(pred, mc)
        print(f"{label:12s}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}"
              f"  Corr={m['corr']:.4f}  Bias={m['bias']:+.4f}")


if __name__ == "__main__":
    main()
