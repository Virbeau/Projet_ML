#!/usr/bin/env python3
"""
Analyse de la distribution du budget par nœud et corrélation avec la betweenness centrality.
"""
import json
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import argparse


def family_from_topology(topology_type):
    if topology_type.startswith("mesh"):
        return "mesh"
    if topology_type.startswith("sp"):
        return "sp"
    if topology_type.startswith("er"):
        return "er"
    return "other"


def extract_node_budgets(instance):
    """
    Extrait le budget alloué à chaque nœud.
    y[i] est pi_star (probabilité de réparation optimale) pour le nœud i.
    Un pi_star élevé signifie qu'on investit plus dans la réparation de ce nœud.
    """
    nodes = instance["graph"]["nodes"]
    y_data = instance["y"]
    B_total = float(instance["B"])
    
    # Budget/effort alloué au nœud = y[i] (pi_star)
    # Plus y[i] est grand, plus on investit dans ce nœud
    node_budgets = {}
    for i, node_id in enumerate(nodes):
        if i < len(y_data):
            node_budgets[node_id] = float(y_data[i])
    
    return node_budgets, B_total


def calculate_betweenness(instance):
    """
    Calcule la betweenness centrality pour chaque nœud du graphe.
    """
    edges = instance["graph"]["edges"]
    is_directed = instance["graph"].get("is_directed", True)
    
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    G.add_edges_from(edges)
    
    try:
        betweenness = nx.betweenness_centrality(G, normalized=True)
    except:
        # Si le graphe est déconnecté ou autre problème
        betweenness = {node: 0.0 for node in G.nodes()}
    
    return betweenness


def analyze_budget_centrality_correlation(instance):
    """
    Analyse la corrélation entre budget alloué (pi_star) et betweenness centrality.
    Retourne des métriques sur l'allocation stratégique du budget.
    """
    node_budgets, B_total = extract_node_budgets(instance)
    betweenness = calculate_betweenness(instance)
    
    # Filtrer les nœuds réparables (ceux qui ont un budget non nul)
    repairable = instance.get("repairable_nodes", [])
    
    if not repairable:
        return None
    
    budgets = []
    centralities = []
    nodes_sorted = []
    
    for node in repairable:
        if node in node_budgets and node in betweenness:
            budgets.append(node_budgets[node])
            centralities.append(betweenness[node])
            nodes_sorted.append(node)
    
    if len(budgets) < 2:
        return None
    
    budgets = np.array(budgets)
    centralities = np.array(centralities)
    
    # Filtrer les cas où tous les budgets sont nuls ou identiques
    if np.std(budgets) < 1e-10 or np.std(centralities) < 1e-10:
        # Budget constant ou centralité constante
        return {
            "n_repairable": len(repairable),
            "B_total": B_total,
            "pearson_corr": 0.0,
            "pearson_p": 1.0,
            "spearman_corr": 0.0,
            "spearman_p": 1.0,
            "mean_budget": float(np.mean(budgets)),
            "std_budget": float(np.std(budgets)),
            "mean_centrality": float(np.mean(centralities)),
            "std_centrality": float(np.std(centralities)),
            "top_budget_efficiency_score": 0.0,
            "budget_concentration": 0.0,
            "budget_active": budgets.sum() > 0,  # Au moins un nœud avec budget > 0
        }
    
    # Corrélation Pearson et Spearman
    try:
        pearson_corr, pearson_p = pearsonr(budgets, centralities)
        if np.isnan(pearson_corr):
            pearson_corr, pearson_p = 0.0, 1.0
    except:
        pearson_corr, pearson_p = 0.0, 1.0
    
    try:
        spearman_corr, spearman_p = spearmanr(budgets, centralities)
        if np.isnan(spearman_corr):
            spearman_corr, spearman_p = 0.0, 1.0
    except:
        spearman_corr, spearman_p = 0.0, 1.0
    
    # Identifier les nœuds avec le plus de budget
    if budgets.sum() > 0:
        top_n = min(3, len(budgets))
        top_budget_indices = np.argsort(budgets)[-top_n:][::-1]
        
        # Score: nœud avec budget élevé ET centralité élevée = bon score
        # Score = budget_normalized * centrality_normalized
        budget_norm = budgets / (budgets.sum() + 1e-10)
        centrality_norm = centralities / (centralities.max() + 1e-10) if centralities.max() > 0 else centralities
        efficiency_scores = budget_norm * centrality_norm
        
        # Moyenne du score pour les top budget nodes
        top_budget_efficiency = np.mean([efficiency_scores[i] for i in top_budget_indices])
        budget_concentration = float(np.std(budget_norm))
    else:
        top_budget_efficiency = 0.0
        budget_concentration = 0.0
    
    results = {
        "n_repairable": len(repairable),
        "B_total": B_total,
        "pearson_corr": float(pearson_corr),
        "pearson_p": float(pearson_p),
        "spearman_corr": float(spearman_corr),
        "spearman_p": float(spearman_p),
        "mean_budget": float(np.mean(budgets)),
        "std_budget": float(np.std(budgets)),
        "mean_centrality": float(np.mean(centralities)),
        "std_centrality": float(np.std(centralities)),
        "top_budget_efficiency_score": float(top_budget_efficiency),
        "budget_concentration": budget_concentration,
        "budget_active": budgets.sum() > 0,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyse budget vs betweenness centrality")
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
        help="Dossier de sortie (défaut: budget_analysis_<nom_dataset>)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"budget_analysis_{dataset_path.stem}")
    out_dir.mkdir(exist_ok=True)
    
    print(f"Chargement du dataset: {dataset_path.name}...")
    with dataset_path.open("r") as f:
        data = json.load(f)
    
    instances = data["instances"]
    
    print(f"Analyse de {len(instances)} instances...")
    
    # Regrouper par topologie et famille
    by_topology = defaultdict(list)
    by_family = defaultdict(list)
    
    for idx, inst in enumerate(instances):
        if idx % 5000 == 0:
            print(f"  Traitement: {idx}/{len(instances)}")
        
        result = analyze_budget_centrality_correlation(inst)
        if result is None:
            continue
        
        topology = inst.get("topology_type", "unknown")
        family = family_from_topology(topology)
        
        by_topology[topology].append(result)
        by_family[family].append(result)
    
    print("Calcul des statistiques agrégées...")
    
    # Statistiques par famille (séparées: avec budget actif vs sans)
    family_stats = {}
    family_stats_active = {}
    
    for fam, results in by_family.items():
        if not results:
            continue
        
        # Toutes les instances
        family_stats[fam] = {
            "count": len(results),
            "count_budget_active": sum(1 for r in results if r.get("budget_active", False)),
            "pearson_mean": float(np.mean([r["pearson_corr"] for r in results])),
            "pearson_std": float(np.std([r["pearson_corr"] for r in results])),
            "spearman_mean": float(np.mean([r["spearman_corr"] for r in results])),
            "spearman_std": float(np.std([r["spearman_corr"] for r in results])),
            "efficiency_score_mean": float(np.mean([r["top_budget_efficiency_score"] for r in results])),
            "efficiency_score_std": float(np.std([r["top_budget_efficiency_score"] for r in results])),
            "budget_concentration_mean": float(np.mean([r["budget_concentration"] for r in results])),
            "budget_concentration_std": float(np.std([r["budget_concentration"] for r in results])),
        }
        
        # Seulement les instances avec budget actif (au moins un nœud avec y > 0)
        results_active = [r for r in results if r.get("budget_active", False)]
        if results_active:
            family_stats_active[fam] = {
                "count": len(results_active),
                "pearson_mean": float(np.mean([r["pearson_corr"] for r in results_active])),
                "pearson_std": float(np.std([r["pearson_corr"] for r in results_active])),
                "spearman_mean": float(np.mean([r["spearman_corr"] for r in results_active])),
                "spearman_std": float(np.std([r["spearman_corr"] for r in results_active])),
                "efficiency_score_mean": float(np.mean([r["top_budget_efficiency_score"] for r in results_active])),
                "budget_concentration_mean": float(np.mean([r["budget_concentration"] for r in results_active])),
            }
    
    # Statistiques par topologie
    topology_stats = {}
    for topo, results in by_topology.items():
        if not results:
            continue
        
        topology_stats[topo] = {
            "count": len(results),
            "pearson_mean": float(np.mean([r["pearson_corr"] for r in results])),
            "pearson_std": float(np.std([r["pearson_corr"] for r in results])),
            "spearman_mean": float(np.mean([r["spearman_corr"] for r in results])),
            "efficiency_score_mean": float(np.mean([r["top_budget_efficiency_score"] for r in results])),
            "budget_concentration_mean": float(np.mean([r["budget_concentration"] for r in results])),
        }
    
    # Sauvegarde JSON
    output_json = out_dir / "budget_centrality_analysis.json"
    with output_json.open("w") as f:
        json.dump({
            "by_family": family_stats,
            "by_family_active_budget_only": family_stats_active,
            "by_topology": topology_stats
        }, f, indent=2)
    
    print(f"✓ Analyse sauvegardée: {output_json}")
    
    # Affichage résumé
    print("\n" + "="*70)
    print("ANALYSE: DISTRIBUTION DU BUDGET vs BETWEENNESS CENTRALITY")
    print("="*70)
    
    print("\nPar FAMILLE (toutes instances):")
    print("-"*70)
    for fam in sorted(family_stats.keys()):
        stats = family_stats[fam]
        pct_active = 100 * stats['count_budget_active'] / stats['count']
        print(f"\n{fam.upper()} ({stats['count']} instances, {pct_active:.1f}% avec budget actif):")
        print(f"  Score d'efficacité (top budget nodes):       {stats['efficiency_score_mean']:.4f} ± {stats['efficiency_score_std']:.4f}")
        print(f"  Concentration du budget (std):               {stats['budget_concentration_mean']:.4f} ± {stats['budget_concentration_std']:.4f}")
    
    print("\n" + "="*70)
    print("Par FAMILLE (budget actif uniquement - au moins un nœud avec y > 0):")
    print("-"*70)
    for fam in sorted(family_stats_active.keys()):
        stats = family_stats_active[fam]
        print(f"\n{fam.upper()} ({stats['count']} instances):")
        print(f"  Corrélation Pearson (budget vs centrality):  {stats['pearson_mean']:+.4f} ± {stats['pearson_std']:.4f}")
        print(f"  Corrélation Spearman:                        {stats['spearman_mean']:+.4f} ± {stats['spearman_std']:.4f}")
        print(f"  Score d'efficacité (top budget nodes):       {stats['efficiency_score_mean']:.4f}")
        print(f"  Concentration du budget (std):               {stats['budget_concentration_mean']:.4f}")
    
    print("\n" + "="*70)
    print("Top 10 topologies - Meilleure allocation stratégique (score efficacité):")
    print("-"*70)
    
    topo_ranked = sorted(topology_stats.items(), 
                         key=lambda x: x[1]["efficiency_score_mean"], 
                         reverse=True)
    
    for i, (topo, stats) in enumerate(topo_ranked[:10], 1):
        print(f"{i:2d}. {topo:20s}  Score: {stats['efficiency_score_mean']:.4f}  "
              f"Corr: {stats['spearman_mean']:+.4f}  (n={stats['count']})")
    
    print("\n" + "="*70)
    print("Top 10 topologies - Pire allocation (score le plus faible):")
    print("-"*70)
    
    for i, (topo, stats) in enumerate(topo_ranked[-10:][::-1], 1):
        print(f"{i:2d}. {topo:20s}  Score: {stats['efficiency_score_mean']:.4f}  "
              f"Corr: {stats['spearman_mean']:+.4f}  (n={stats['count']})")
    
    # Génération des visualisations
    print("\nGénération des visualisations...")
    generate_visualizations(family_stats, family_stats_active, topology_stats, out_dir)
    
    print("\n✓ Analyse terminée!")
    print(f"  Résultats JSON: {output_json}")
    print(f"  Visualisations: {out_dir}/")


def generate_visualizations(family_stats, family_stats_active, topology_stats, out_dir):
    """Génère des visualisations de l'analyse budget-centralité."""
    
    # Figure 1: Corrélations par famille (budget actif seulement)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    families = sorted(family_stats_active.keys())
    pearson_means = [family_stats_active[f]["pearson_mean"] for f in families]
    pearson_stds = [family_stats_active[f]["pearson_std"] for f in families]
    spearman_means = [family_stats_active[f]["spearman_mean"] for f in families]
    efficiency_means = [family_stats_active[f]["efficiency_score_mean"] for f in families]
    
    colors = {"mesh": "#66c2a5", "sp": "#fc8d62", "er": "#8da0cb"}
    bar_colors = [colors.get(f, "#999999") for f in families]
    
    # Sous-graphe 1: Corrélation Pearson
    axes[0].bar(families, pearson_means, yerr=pearson_stds, 
                color=bar_colors, alpha=0.8, capsize=5)
    axes[0].set_title("Corrélation Pearson\n(Budget vs Betweenness)")
    axes[0].set_ylabel("Corrélation")
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Sous-graphe 2: Corrélation Spearman
    axes[1].bar(families, spearman_means, color=bar_colors, alpha=0.8)
    axes[1].set_title("Corrélation Spearman\n(Budget vs Betweenness)")
    axes[1].set_ylabel("Corrélation")
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Sous-graphe 3: Score d'efficacité
    axes[2].bar(families, efficiency_means, color=bar_colors, alpha=0.8)
    axes[2].set_title("Score d'efficacité\n(Top budget nodes)")
    axes[2].set_ylabel("Score")
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "01_budget_centrality_by_family.png", 
                dpi=200, bbox_inches="tight")
    plt.close()
    
    # Figure 2: Topologies triées par score d'efficacité
    topo_sorted = sorted(topology_stats.items(), 
                         key=lambda x: x[1]["efficiency_score_mean"], 
                         reverse=True)
    
    top_15 = topo_sorted[:15]
    names = [t[0] for t in top_15]
    scores = [t[1]["efficiency_score_mean"] for t in top_15]
    corrs = [t[1]["spearman_mean"] for t in top_15]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    y_pos = np.arange(len(names))
    
    axes[0].barh(y_pos, scores, color="#2ca02c", alpha=0.8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Score d'efficacité")
    axes[0].set_title("Top 15 topologies - Meilleure allocation budget/centralité")
    axes[0].grid(axis='x', alpha=0.3)
    
    axes[1].barh(y_pos, corrs, color="#d62728", alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Corrélation Spearman")
    axes[1].set_title("Corrélation budget-centralité (Top 15)")
    axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "02_top_topologies_budget_efficiency.png", 
                dpi=200, bbox_inches="tight")
    plt.close()
    
    # Figure 3: Concentration du budget par famille (toutes instances)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    families_all = sorted(family_stats.keys())
    concentration_means = [family_stats[f]["budget_concentration_mean"] for f in families_all]
    concentration_stds = [family_stats[f]["budget_concentration_std"] for f in families_all]
    colors_all = {"mesh": "#66c2a5", "sp": "#fc8d62", "er": "#8da0cb"}
    bar_colors_all = [colors_all.get(f, "#999999") for f in families_all]
    
    ax.bar(families_all, concentration_means, yerr=concentration_stds, 
           color=bar_colors_all, alpha=0.8, capsize=5)
    ax.set_title("Concentration du budget entre les nœuds\n(plus élevé = budget concentré sur peu de nœuds)")
    ax.set_ylabel("Std de la distribution normalisée")
    ax.set_xlabel("Famille de topologie")
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "03_budget_concentration_by_family.png", 
                dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ 3 visualisations générées dans {out_dir}/")


if __name__ == "__main__":
    main()
