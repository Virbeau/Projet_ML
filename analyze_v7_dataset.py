import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILES = {
    "er": "datasetV7_er.json",
    "mesh": "datasetV7_mesh.json",
    "sp": "datasetV7_sp.json",
}
OUT_DIR = Path("analysis_v7")


def load_instances(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("instances", [])


def build_dataframe():
    rows = []
    for family, path in INPUT_FILES.items():
        if not os.path.exists(path):
            continue
        instances = load_instances(path)
        for inst in instances:
            c_total = float(inst.get("C_total", 0.0))
            c_min = float(inst.get("C_min_path", 0.0))
            n_nodes = int(inst.get("n_nodes", 0))
            n_edges = int(inst.get("n_edges", 0))
            max_edges = max(1, n_nodes * max(1, n_nodes - 1))
            n_rep = len(inst.get("repairable_nodes", []))
            b = float(inst.get("B", 0.0))
            j = float(inst.get("J_star", np.nan))

            rows.append(
                {
                    "family": family,
                    "topology_type": inst.get("topology_type", ""),
                    "J_star": j,
                    "B": b,
                    "C_total": c_total,
                    "C_min_path": c_min,
                    "budget_ratio_total": b / c_total if c_total > 1e-12 else np.nan,
                    "budget_ratio_path": b / c_min if c_min > 1e-12 else np.nan,
                    "alpha": float(inst.get("alpha", np.nan)),
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "density": n_edges / max_edges,
                    "n_repairable": n_rep,
                    "shortest_path_length": float(inst.get("shortest_path_length", np.nan)),
                    "H": float(inst.get("H", np.nan)),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Aucune instance V7 trouvée. Vérifiez les fichiers datasetV7_*.json")
    return df


def fit_ols(y, X):
    X_ = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
    y_hat = X_ @ beta
    resid = y - y_hat
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    return beta, y_hat, resid, r2


def standardize_cols(df, cols):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        mu = df[c].mean()
        sd = df[c].std()
        out[c] = (df[c] - mu) / (sd if sd > 1e-12 else 1.0)
    return out


def compute_budget_vs_topology_effect(df):
    dfm = df.dropna(
        subset=[
            "J_star",
            "budget_ratio_total",
            "n_nodes",
            "n_edges",
            "density",
            "n_repairable",
            "shortest_path_length",
            "C_total",
        ]
    ).copy()

    fam_dummies = pd.get_dummies(dfm["family"], drop_first=True)
    difficulty_cols = [
        "n_nodes",
        "n_edges",
        "density",
        "n_repairable",
        "shortest_path_length",
        "C_total",
    ]
    Zdiff = standardize_cols(dfm, difficulty_cols)
    Zbudget = standardize_cols(dfm, ["budget_ratio_total"])

    y = dfm["J_star"].values

    X0 = fam_dummies.values
    _, _, _, r2_0 = fit_ols(y, X0)

    X1 = np.column_stack([fam_dummies.values, Zdiff.values])
    _, _, _, r2_1 = fit_ols(y, X1)

    X2 = np.column_stack([fam_dummies.values, Zdiff.values, Zbudget.values])
    b2, yhat2, resid2, r2_2 = fit_ols(y, X2)

    X3 = np.column_stack([Zdiff.values, Zbudget.values])
    _, _, _, r2_3 = fit_ols(y, X3)

    # Partial effect budget: residual-residual plot components
    _, _, ry, _ = fit_ols(y, np.column_stack([fam_dummies.values, Zdiff.values]))
    x_budget = Zbudget["budget_ratio_total"].values
    _, _, rx, _ = fit_ols(x_budget, np.column_stack([fam_dummies.values, Zdiff.values]))

    return {
        "dfm": dfm,
        "ry": ry,
        "rx": rx,
        "r2_family_only": r2_0,
        "r2_family_difficulty": r2_1,
        "r2_full": r2_2,
        "r2_no_family": r2_3,
        "delta_r2_budget": r2_2 - r2_1,
        "delta_r2_topology": r2_2 - r2_3,
        "beta_budget_std": float(b2[-1]),
    }


def make_plots(df, effect):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Family composition
    plt.figure(figsize=(7, 4))
    counts = df["family"].value_counts().sort_index()
    counts.plot(kind="bar", color=["#3b82f6", "#10b981", "#f59e0b"])
    plt.title("Composition du dataset V7 par famille")
    plt.ylabel("Nombre d'instances")
    plt.xlabel("Famille")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_family_counts.png", dpi=160)
    plt.close()

    # 2) J* distribution by family
    plt.figure(figsize=(8, 5))
    fam_order = ["er", "sp", "mesh"]
    data = [df.loc[df["family"] == f, "J_star"].dropna().values for f in fam_order]
    plt.boxplot(data, tick_labels=fam_order, showfliers=False)
    plt.title("Distribution de J* par famille")
    plt.ylabel("J* (downtime moyen)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_jstar_boxplot_family.png", dpi=160)
    plt.close()

    # 3) B vs J* (raw)
    plt.figure(figsize=(8, 5))
    for fam, col in [("er", "#3b82f6"), ("sp", "#10b981"), ("mesh", "#f59e0b")]:
        d = df[df["family"] == fam]
        plt.scatter(d["B"], d["J_star"], s=8, alpha=0.25, label=fam, c=col)
    plt.title("Relation brute Budget B vs J*")
    plt.xlabel("B")
    plt.ylabel("J*")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_scatter_B_vs_J.png", dpi=160)
    plt.close()

    # 4) Budget ratio vs J*
    plt.figure(figsize=(8, 5))
    for fam, col in [("er", "#3b82f6"), ("sp", "#10b981"), ("mesh", "#f59e0b")]:
        d = df[df["family"] == fam]
        plt.scatter(d["budget_ratio_total"], d["J_star"], s=8, alpha=0.25, label=fam, c=col)
    plt.title("Relation budget normalisé (B/C_total) vs J*")
    plt.xlabel("B / C_total")
    plt.ylabel("J*")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_scatter_ratio_vs_J.png", dpi=160)
    plt.close()

    # 5) Binned trend by family
    plt.figure(figsize=(8, 5))
    for fam, col in [("er", "#3b82f6"), ("sp", "#10b981"), ("mesh", "#f59e0b")]:
        d = df[df["family"] == fam].dropna(subset=["budget_ratio_total", "J_star"]).copy()
        d["bin"] = pd.qcut(d["budget_ratio_total"], q=10, duplicates="drop")
        g = d.groupby("bin", observed=True).agg(ratio_mean=("budget_ratio_total", "mean"), j_mean=("J_star", "mean"))
        plt.plot(g["ratio_mean"], g["j_mean"], marker="o", linewidth=2, label=fam, color=col)
    plt.title("Effet moyen du budget normalisé sur J* (par déciles)")
    plt.xlabel("Budget normalisé moyen (B/C_total)")
    plt.ylabel("J* moyen")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_binned_ratio_effect.png", dpi=160)
    plt.close()

    # 6) Correlation heatmap (global)
    cols = ["J_star", "B", "budget_ratio_total", "budget_ratio_path", "n_nodes", "n_edges", "density", "n_repairable", "shortest_path_length"]
    corr = df[cols].corr(method="spearman")
    plt.figure(figsize=(9, 7))
    plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.colorbar(label="Spearman")
    plt.title("Corrélations Spearman (global)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_correlation_heatmap.png", dpi=160)
    plt.close()

    # 7) Partial budget effect (controlled)
    rx = effect["rx"]
    ry = effect["ry"]
    plt.figure(figsize=(8, 5))
    plt.scatter(rx, ry, s=8, alpha=0.25, c="#374151")
    p = np.polyfit(rx, ry, deg=1)
    xx = np.linspace(np.min(rx), np.max(rx), 100)
    plt.plot(xx, p[0] * xx + p[1], color="#dc2626", linewidth=2)
    plt.title("Effet partiel du budget sur J* (controle topologie + difficulte)")
    plt.xlabel("Residual budget_ratio_total")
    plt.ylabel("Residual J*")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_partial_budget_effect.png", dpi=160)
    plt.close()


def write_summary(df, effect):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "SUMMARY_V7.md"

    fam_stats = (
        df.groupby("family", observed=True)
        .agg(
            n_instances=("J_star", "size"),
            j_mean=("J_star", "mean"),
            j_std=("J_star", "std"),
            B_mean=("B", "mean"),
            ratio_mean=("budget_ratio_total", "mean"),
            nodes_mean=("n_nodes", "mean"),
            edges_mean=("n_edges", "mean"),
        )
        .sort_index()
    )

    corr_global = df[["J_star", "B", "budget_ratio_total", "budget_ratio_path", "n_nodes", "n_edges", "density", "n_repairable"]].corr(method="spearman")
    corr_by_family = {}
    for fam in sorted(df["family"].unique()):
        d = df[df["family"] == fam]
        corr_by_family[fam] = d[["J_star", "B", "budget_ratio_total", "budget_ratio_path", "n_nodes", "n_edges", "density", "n_repairable"]].corr(method="spearman")

    lines = []
    lines.append("# Analyse V7 - Composition et qualite")
    lines.append("")
    lines.append("## Donnees analysees")
    lines.append(f"- Nombre total d'instances: {len(df)}")
    lines.append(f"- Familles: {', '.join(sorted(df['family'].unique()))}")
    lines.append("")
    lines.append("## Statistiques par famille")
    lines.append("```")
    lines.append(fam_stats.to_string(float_format=lambda x: f"{x:.4f}"))
    lines.append("```")
    lines.append("")
    lines.append("## Correlations globales (Spearman)")
    lines.append("```")
    lines.append(corr_global.to_string(float_format=lambda x: f"{x:.4f}"))
    lines.append("```")
    lines.append("")
    lines.append("## Correlations budget -> J* par famille (Spearman)")
    for fam in sorted(corr_by_family.keys()):
        c = corr_by_family[fam]
        lines.append(
            f"- {fam}: corr(J*, B)={c.loc['J_star','B']:.4f}, "
            f"corr(J*, B/C_total)={c.loc['J_star','budget_ratio_total']:.4f}, "
            f"corr(J*, B/C_min_path)={c.loc['J_star','budget_ratio_path']:.4f}"
        )
    lines.append("")
    lines.append("## Influence budget vs topologie (controle de difficulte)")
    lines.append(
        "- Modele R2 famille seule: "
        f"{effect['r2_family_only']:.4f}"
    )
    lines.append(
        "- Modele R2 famille + difficulte: "
        f"{effect['r2_family_difficulty']:.4f}"
    )
    lines.append(
        "- Modele R2 famille + difficulte + budget_ratio_total: "
        f"{effect['r2_full']:.4f}"
    )
    lines.append(
        "- Gain R2 du budget (au-dela topologie+difficulte): "
        f"{effect['delta_r2_budget']:.4f}"
    )
    lines.append(
        "- Gain R2 attribuable a la topologie (au-dela difficulte+budget): "
        f"{effect['delta_r2_topology']:.4f}"
    )
    lines.append(
        "- Coefficient standardise budget_ratio_total (modele complet): "
        f"{effect['beta_budget_std']:.4f}"
    )
    lines.append("")
    lines.append("## Lecture rapide")
    lines.append("- Si le coefficient budget est negatif, augmenter le budget fait baisser J* (a difficulte/topologie comparables).")
    lines.append("- Le gain R2 du budget quantifie l'impact marginal du budget independamment de la difficulte et de la famille.")
    lines.append("- Le gain R2 de la topologie quantifie l'effet structurel propre a la famille de graphe.")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataframe()
    effect = compute_budget_vs_topology_effect(df)
    make_plots(df, effect)
    write_summary(df, effect)
    df.to_csv(OUT_DIR / "v7_flat_table.csv", index=False)

    print("Analyse V7 terminee")
    print(f"- Dossier: {OUT_DIR}")
    print(f"- Instances analysees: {len(df)}")
    print(f"- Gain R2 budget (controle): {effect['delta_r2_budget']:.4f}")
    print(f"- Gain R2 topologie (controle): {effect['delta_r2_topology']:.4f}")
    print(f"- Beta budget standardise: {effect['beta_budget_std']:.4f}")


if __name__ == "__main__":
    main()
