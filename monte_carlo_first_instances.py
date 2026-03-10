#!/usr/bin/env python3
"""
Script pour valider et enregistrer les résultats Monte-Carlo 
pour les premières instances d'un dataset.

Usage:
    python monte_carlo_first_instances.py --first-n 10
    python monte_carlo_first_instances.py --dataset dataset_hybrid_mesh_sp_er.json --first-n 5
"""

import json
import argparse
from pathlib import Path
from main_monte_carlo_validation import validate_dataset, save_results, print_results

def validate_first_instances(dataset_path, first_n=10, n_sims=10000):
    """
    Valide les N premières instances d'un dataset et enregistre les résultats.
    
    Args:
        dataset_path: Chemin vers le dataset
        first_n: Nombre de premières instances à valider
        n_sims: Nombre de simulations Monte-Carlo par instance
    """
    print(f"\n🔍 Chargement du dataset: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    all_instances = data["instances"]
    print(f"   Total d'instances dans le dataset: {len(all_instances)}")
    
    if first_n > len(all_instances):
        print(f"   ⚠️  Seules {len(all_instances)} instances disponibles (au lieu de {first_n})")
        first_n = len(all_instances)
    
    print(f"   Validation des {first_n} premières instances...")
    print(f"   Simulations par instance: {n_sims}\n")
    
    # Créer une copie temporaire du dataset avec seulement les N premières instances
    temp_data = data.copy()
    temp_data["instances"] = all_instances[:first_n]
    
    # Créer un fichier temporaire
    temp_path = Path(dataset_path).stem + f"_first_{first_n}_temp.json"
    with open(temp_path, 'w') as f:
        json.dump(temp_data, f)
    try:
        # Lancer la validation
        results = validate_dataset(
            temp_path,
            sample_size=None,  # Utiliser toutes les instances du dataset temp
            n_sims=n_sims,
            verbose=True
        )
        
        # Afficher les résultats
        print_results(results)
        
        # Sauvegarder les résultats avec un nom explicite
        timestamp = Path(save_results(results)).stem
        output_filename = f"mc_validation_first_{first_n}_{timestamp}.json"
        
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Résultats sauvegardés: {output_filename}\n")
        
        return results, output_filename
        
    finally:
        # Nettoyer le fichier temporaire
        if Path(temp_path).exists():
            Path(temp_path).unlink()

def main():
    parser = argparse.ArgumentParser(
        description="Valider et enregistrer les résultats Monte-Carlo pour les premières instances"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_hybrid_mesh_sp_er.json",
        help="Chemin vers le dataset (défaut: dataset_hybrid_mesh_sp_er.json)"
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=10,
        help="Nombre de premières instances à valider (défaut: 10)"
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=10000,
        help="Nombre de simulations Monte-Carlo par instance (défaut: 10000)"
    )
    
    args = parser.parse_args()
    
    # Vérifier que le dataset existe
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Erreur: Le dataset '{args.dataset}' n'existe pas")
        print(f"\n   Fichiers disponibles:")
        for f in Path(".").glob("dataset*.json"):
            print(f"   - {f}")
        return 1
    
    # Lancer la validation des premières instances
    try:
        results, output_file = validate_first_instances(
            dataset_path,
            first_n=args.first_n,
            n_sims=args.n_sims
        )
        
        # Résumé des résultats
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ")
        print("=" * 60)
        print(f"Instances validées: {args.first_n}")
        print(f"MAE (Erreur Absolue Moyenne): {results['statistics']['MAE']:.6f}")
        print(f"Fichier de sortie: {output_file}")
        print("=" * 60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
