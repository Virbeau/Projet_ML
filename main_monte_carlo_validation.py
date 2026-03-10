#!/usr/bin/env python3
"""
Script principal pour la validation Monte-Carlo des résultats.
Permet de valider la précision des valeurs J* calculées analytiquement
en les comparant à une simulation Monte-Carlo.
"""

import json
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import argparse

def fast_is_connected(G_base, source, target, up_nodes):
    """
    Vérifie la connectivité sur le sous-graphe des noeuds fonctionnels.
    """
    # Les terminaux sont toujours UP
    valid_nodes = set(up_nodes) | {source, target}
    subG = G_base.subgraph(valid_nodes)
    try:
        return nx.has_path(subG, source, target)
    except nx.NetworkXPointlessConcept:
        return False

def simulate_monte_carlo(instance, n_sims=10000):
    """
    Simule n_sims trajectoires markoviennes vectorisées pour un graphe donné.
    
    Args:
        instance: Instance du dataset contenant le graphe et ses paramètres
        n_sims: Nombre de simulations à effectuer
        
    Returns:
        empirical_J: Probabilité empirique de défaillance
    """
    H = int(instance["H"])
    edges = instance["graph"]["edges"]
    nodes = instance["graph"]["nodes"]  # Récupération de la liste complète des noeuds
    terminals = instance["terminals"]
    source, target = terminals[0], terminals[1]
    
    rep_nodes = instance.get("repairable_nodes", [])
    if not rep_nodes:
        rep_nodes = [n for n in nodes if n not in terminals]
    
    # CORRECTION: Mapping entre ID du noeud et son index dans la liste x/y
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    # Extraire p_fail et pi_star avec le bon mapping
    p_fail = np.array([instance["x"][node_to_idx[n]][0] for n in rep_nodes])
    pi_star = np.array([instance["y"][node_to_idx[n]] for n in rep_nodes])
    
    # Créer le graphe de base statique
    G = nx.DiGraph() if instance["graph"].get("is_directed", True) else nx.Graph()
    G.add_edges_from(edges)
    
    # Matrice d'états (n_sims, n_repairable) - Initialement, tout le monde est UP
    states = np.ones((n_sims, len(rep_nodes)), dtype=bool)
    
    # Suivi des simulations qui ont déjà échoué (état absorbant)
    failed_sims = np.zeros(n_sims, dtype=bool)
    
    for step in range(H):
        # 1. Mise à jour probabiliste des états (Markov)
        rands = np.random.rand(n_sims, len(rep_nodes))
        
        # Ce qui était UP tombe en panne si rand < p_fail
        to_down = states & (rands < p_fail)
        # Ce qui était DOWN est réparé si rand < pi_star
        to_up = (~states) & (rands < pi_star)
        
        # Nouvel état
        states = (states & ~to_down) | to_up
        
        # 2. Vérification topologique uniquement pour les systèmes encore en vie
        active_indices = np.where(~failed_sims)[0]
        
        for idx in active_indices:
            # Mapping des noeuds réparables qui sont UP dans cette simulation
            up_rep_nodes = [rep_nodes[i] for i, is_up in enumerate(states[idx]) if is_up]
            
            # Si plus de chemin, le système meurt (état absorbant)
            if not fast_is_connected(G, source, target, up_rep_nodes):
                failed_sims[idx] = True
                
        # Optimisation : si tout le monde a échoué avant H, on arrête
        if failed_sims.all():
            break

    # La probabilité de défaillance est le ratio de simulations ayant échoué
    empirical_J = failed_sims.sum() / n_sims
    return empirical_J

def validate_dataset(dataset_path, sample_size=None, n_sims=10000, verbose=True):
    """
    Lance la validation Monte-Carlo sur un dataset.
    
    Args:
        dataset_path: Chemin vers le fichier JSON du dataset
        sample_size: Nombre d'instances à valider (None = tout le dataset)
        n_sims: Nombre de simulations Monte-Carlo par instance
        verbose: Afficher les détails
        
    Returns:
        dict: Résultats de la validation
    """
    
    if verbose:
        print(f"Chargement du dataset: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    all_instances = data["instances"]
    metadata = data.get("metadata", {})
    
    # Déterminer la taille d'échantillonnage
    if sample_size is None:
        sample_size = len(all_instances)
    else:
        sample_size = min(sample_size, len(all_instances))
    
    if sample_size < len(all_instances):
        validation_sample = random.sample(all_instances, sample_size)
    else:
        validation_sample = all_instances
    
    if verbose:
        print(f"Nombre d'instances à valider: {sample_size}/{len(all_instances)}")
        print(f"Simulations par instance: {n_sims}")
        print("=" * 60)
    
    errors = []
    results_detail = []
    
    for idx, inst in enumerate(tqdm(validation_sample, desc="Validation", disable=not verbose)):
        j_emp = simulate_monte_carlo(inst, n_sims=n_sims)
        j_ana = inst["J_star"]
        
        # Calcul de l'erreur absolue et relative
        error_abs = abs(j_emp - j_ana)
        error_rel = error_abs / j_ana if j_ana > 0 else 0.0
        errors.append(error_abs)
        
        results_detail.append({
            "instance_id": idx,
            "topology": inst.get("topology", "unknown"),
            "J_analytical": j_ana,
            "J_empirical": j_emp,
            "error_absolute": error_abs,
            "error_relative": error_rel
        })
        
        # Affichage de debug tous les 500 graphes
        if verbose and idx % 500 == 0 and idx > 0:
            mae_so_far = np.mean(errors)
            tqdm.write(f"Graphe {idx}: J*={j_ana:.4f} | MC={j_emp:.4f} | Erreur={error_abs:.4f} | MAE={mae_so_far:.4f}")

    # Statistiques finales
    errors_array = np.array(errors)
    mae = np.mean(errors_array)
    mse = np.mean(errors_array ** 2)
    rmse = np.sqrt(mse)
    std = np.std(errors_array)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_metadata": metadata,
        "validation_config": {
            "sample_size": sample_size,
            "n_simulations": n_sims,
            "total_instances": len(all_instances)
        },
        "statistics": {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "STD": float(std),
            "min_error": float(errors_array.min()),
            "max_error": float(errors_array.max()),
            "median_error": float(np.median(errors_array))
        },
        "details": results_detail
    }
    
    return results

def save_results(results, output_path=None):
    """
    Sauvegarde les résultats de la validation.
    
    Args:
        results: Dict des résultats
        output_path: Chemin de sortie (None = auto-généré)
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"validation_results_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return output_path

def print_results(results):
    """Affiche les résultats de la validation."""
    print("\n" + "=" * 60)
    print("RÉSULTATS DE LA VALIDATION MONTE-CARLO")
    print("=" * 60)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Dataset: {results['dataset_path']}")
    print(f"Instances validées: {results['validation_config']['sample_size']}/{results['validation_config']['total_instances']}")
    print(f"Simulations par instance: {results['validation_config']['n_simulations']}")
    print("-" * 60)
    print("STATISTIQUES:")
    stats = results['statistics']
    print(f"  Erreur Absolue Moyenne (MAE):      {stats['MAE']:.6f}")
    print(f"  Erreur Quadratique Moyenne (MSE):  {stats['MSE']:.6f}")
    print(f"  Racine MSE (RMSE):                 {stats['RMSE']:.6f}")
    print(f"  Écart-type:                        {stats['STD']:.6f}")
    print(f"  Min erreur:                        {stats['min_error']:.6f}")
    print(f"  Max erreur:                        {stats['max_error']:.6f}")
    print(f"  Médiane erreur:                    {stats['median_error']:.6f}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Validation Monte-Carlo des résultats analytiques"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_hybrid_mesh_sp_er_v2_1000.json",
        help="Chemin vers le dataset à valider (défaut: dataset_hybrid_mesh_sp_er_v2_1000.json)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Nombre d'instances à valider (défaut: tout le dataset)"
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=10000,
        help="Nombre de simulations Monte-Carlo par instance (défaut: 10000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Chemin pour sauvegarder les résultats (défaut: auto-généré)"
    )
    
    args = parser.parse_args()
    
    # Vérifier que le dataset existe
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Erreur: Le dataset '{args.dataset}' n'existe pas")
        return 1
    
    # Lancer la validation
    print("\n🚀 Démarrage de la validation Monte-Carlo...\n")
    results = validate_dataset(
        dataset_path,
        sample_size=args.sample_size,
        n_sims=args.n_sims,
        verbose=True
    )
    
    # Afficher les résultats
    print_results(results)
    
    # Sauvegarder les résultats
    output_file = save_results(results, args.output)
    print(f"\n✓ Résultats sauvegardés: {output_file}\n")
    
    return 0

if __name__ == "__main__":
    exit(main())
