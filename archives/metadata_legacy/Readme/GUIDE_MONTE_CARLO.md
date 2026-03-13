# Guide de Validation Monte-Carlo

Ce guide explique comment utiliser les scripts de validation Monte-Carlo pour valider la précision de vos résultats analytiques.

## Vue d'ensemble

Les scripts effectuent une **validation Monte-Carlo** en simulant les trajectoires markoviennes de vos graphes et en comparant les résultats empiriques aux valeurs analytiques calculées (J*).

### Erreurs mesurées

- **MAE (Mean Absolute Error)**: Erreur absolue moyenne entre J* analytique et J empirique
- **MSE (Mean Squared Error)**: 
- **RMSE**: Racine de MSE
- **Détails par instance**: Liste complète des erreurs avec topologies

## Installation des dépendances

```bash
cd /users/local/l24virbe/Projet_ML
source env_projet/bin/activate
pip install -r requirements.txt
```

## Scripts disponibles

### 1. **monte_carlo_validation.py** (Simple & Rapide)

Script autonome idéal pour les tests et déploiement rapide.

**Syntaxe:**
```bash
source env_projet/bin/activate
python3 monte_carlo_validation.py
```

**Comportement:**
- Utilise **dataset_hybrid_mesh_sp_er_100.json** par défaut (100 instances)
- Effectue **10,000 simulations** par graphe
- Affiche les résultats dans le terminal
- ⏱️ Durée estimée: 15-30 minutes pour 100 instances

**Exemple de sortie:**
```
Chargement du dataset...
Démarrage de la simulation Monte-Carlo sur 100/100 graphes...
Chaque graphe sera simulé 10,000 fois.
============================================================
Graphe 500 | J*: 0.8234 | MC: 0.8156 | Erreur: 0.0078 | MAE: 0.0234
...
============================================================
VALIDATION MONTE-CARLO TERMINÉE
Instances validées: 100/100
Erreur Absolue Moyenne (MAE):      0.023456
...
```

### 2. **main_monte_carlo_validation.py** (Avancé & Flexible)

Script principal modulable avec options de ligne de commande et sauvegarde JSON.

**Syntaxe de base:**
```bash
source env_projet/bin/activate
python3 main_monte_carlo_validation.py --dataset <dataset> [options]
```

**Options disponibles:**

| Option | Par défaut | Description |
|--------|-----------|-------------|
| `--dataset` | dataset_hybrid_mesh_sp_er_100.json | Chemin du dataset |
| `--sample-size` | Tout le dataset | Nombre d'instances à valider |
| `--n-sims` | 10000 | Simulations par instance |
| `--output` | Auto-généré | Fichier de sortie JSON |

**Exemples d'utilisation:**

```bash
# Validation rapide: 50 instances, 1000 simulations chacune
python3 main_monte_carlo_validation.py --dataset dataset_hybrid_mesh_sp_er_100.json \
  --sample-size 50 --n-sims 1000

# Dataset complet (50k): 100 instances aléatoires pour tester
python3 main_monte_carlo_validation.py --dataset dataset_hybrid_mesh_sp_er.json \
  --sample-size 100

# Haute précision: 100 instances, 50000 simulations
python3 main_monte_carlo_validation.py --dataset dataset_hybrid_mesh_sp_er_100.json \
  --n-sims 50000 --output results_high_precision.json

# Output personnalisé
python3 main_monte_carlo_validation.py --output my_validation_results.json
```

**Sortie JSON:**

Les résultats sont sauvegardés dans un fichier JSON avec la structure:
```json
{
  "timestamp": "2026-03-03T10:30:45.123456",
  "dataset_path": "dataset_hybrid_mesh_sp_er_100.json",
  "validation_config": {
    "sample_size": 100,
    "n_simulations": 10000,
    "total_instances": 100
  },
  "statistics": {
    "MAE": 0.0356,
    "MSE": 0.0021,
    "RMSE": 0.0458,
    "STD": 0.0234,
    "min_error": 0.0001,
    "max_error": 0.1234,
    "median_error": 0.0234
  },
  "details": [
    {
      "instance_id": 0,
      "topology": "mesh_2x2",
      "J_analytical": 0.9998,
      "J_empirical": 0.9856,
      "error_absolute": 0.0142,
      "error_relative": 0.0142
    },
    ...
  ]
}
```

## Recommandations d'utilisation

### Cas d'usage 1: Validation rapide (5-10 min)
```bash
python3 main_monte_carlo_validation.py --sample-size 20 --n-sims 1000
```

### Cas d'usage 2: Validation standard (30-60 min)
```bash
python3 main_monte_carlo_validation.py --sample-size 100 --n-sims 5000
```

### Cas d'usage 3: Validation exhaustive (plusieurs heures)
```bash
python3 main_monte_carlo_validation.py --n-sims 50000
```

### Cas d'usage 4: Dataset complet
```bash
# Valider 1000 instances aléatoires du dataset 50k
python3 main_monte_carlo_validation.py --dataset dataset_hybrid_mesh_sp_er.json \
  --sample-size 1000 --n-sims 10000 --output full_dataset_validation.json
```

## Interprétation des résultats

### Métriques de qualité

- **MAE < 0.01**: Excellente concordance analytique ✓
- **MAE 0.01-0.05**: Bonne concordance ✓
- **MAE 0.05-0.1**: Concordance acceptable
- **MAE > 0.1**: À investiguer

### Facteurs affectant la précision

1. **Nombre de simulations**: Plus = plus précis (mais plus lent)
   - 1,000: ±10-20% erreur statistique
   - 10,000: ±3-5% erreur statistique
   - 50,000: ±1-2% erreur statistique

2. **Type de graphe**: Certains graphes sont plus sensibles que d'autres
   - Graphes très fiables (J* proche de 0): Plus difficiles
   - Graphes peu fiables (J* proche de 1): Plus faciles

3. **Horizon H**: Plus H est grand, plus la durée augmente

## Dépannage

### Erreur: "ModuleNotFoundError: No module named 'tqdm'"

**Solution:**
```bash
source env_projet/bin/activate
pip install tqdm
```

### Erreur: "FileNotFoundError: [Errno 2] No such file or directory: 'dataset_..json'"

**Solution:** Vérifiez que le dataset existe dans le répertoire courant
```bash
ls -la *.json
```

### Script très lent

**Solutions:**
- Réduire `--sample-size`:  `-sample-size 20`
- Réduire `--n-sims`: `--n-sims 1000`
- Utiliser un dataset plus petit

### Résultats incohérents

Possible si:
1. J* pour certains graphes est très proche de 0 ou 1 (difficiles à estimer)
2. H est très grand (long horizon markovien)
3. Trop peu de simulations (augmentez `--n-sims`)

## Performance

Sur une machine standard:
- **Monte-Carlo simple** (1 graphe, 10k simulations): ~3-5 secondes
- **100 graphes x 10k sims**: 15-30 minutes
- **1000 graphes x 10k sims**: 3-5 heures

## Intégration dans votre pipeline

```python
# Dans votre code
from main_monte_carlo_validation import validate_dataset, print_results

results = validate_dataset(
    "dataset_hybrid_mesh_sp_er_100.json",
    sample_size=50,
    n_sims=10000
)
print_results(results)
```

## Support

Pour toute question ou problème, consultez:
- `monte_carlo_validation.py`: Implémentation de la simulation
- `main_monte_carlo_validation.py`: Interface principale
- Logs de validatio dans le fichier JSON de sortie
