# 📊 Guide de Génération 50 000 Instances

## ⚡ Estimation des temps

Basé sur le test avec **100 instances = 2.39s/graphe** :

| Instances | Temps estimé | Débit |
|-----------|-------------|-------|
| 100       | ~4 min      | 1500 graphes/h |
| 1 000     | ~40 min     | 1500 graphes/h |
| 10 000    | ~6.7 h      | 1500 graphes/h |
| **50 000**| **~33 h**   | **1500 graphes/h** |

⚠️ **Note** : Les graphes larges (10-12 nœuds) sont plus lents. La distribution pondérée (70% petits, 25% moyens, 5% gros) optimize la vitesse.

---

## 🚀 Lancer la génération complète

### **Option 1 : Laisser tourner le terminal (simple)**
```bash
cd /users/local/l24virbe/Projet_ML
source env_projet/bin/activate
python main_production.py
```

### **Option 2 : Exécution en arrière-plan (recommandé)**

#### Sur Linux/Mac :
```bash
nohup python main_production.py > generation.out 2>&1 &
```

#### Puis suivre la progression :
```bash
tail -f generation.out           # Temps réel
tail -f generation_progress.log  # Bilan complet
```

---

## 📋 Paramètres à modifier

Dans [main_production.py](main_production.py#L138) (ligne 138) :

```python
n_instances = 50000  # À changer de 100 à 50 000
```

**Autres paramètres optionnels** :

```python
# Ligne 133 : Nombre de workers (4 pour 4 CPU cores)
n_workers = min(4, cpu_count(), n_instances)  # Modifier le 4

# Ligne 119 : Plages de variabilité
H_min, H_max = 5, 20      # Horizon
B_min, B_max = 1.0, 5.0   # Budget
```

---

## 📊 Fichiers générés

### **Dataset principal**
- `dataset_hybrid_mesh_sp_er.json` (~250 MB pour 50k)
- Contient métadonnées + 50 000 instances GNN-ready

### **Fichier log**
- `generation_progress.log` : résumé de la génération
- Timestamps, temps écoulé, débit, statistiques finales

---

## 🔍 Suivi en temps réel

### **Barre de progression (tqdm)**
```
Génération: 45%|███████▎                    | 22500/50000 [5h 30m<6h 45m, 1.50s/graph]
```

### **Statistiques affichées en live**
- Progression / Total
- Temps écoulé / Estimé
- Débit (graphes/heure)

### **Log texte automatique**
- Horodatage du démarrage/fin
- Répartition MESH/SP/ER
- Temps moyen par graphe
- Résultats finaux

---

## ⚙️ Post-génération

### **Vérifier l'intégrité**
```bash
python -c "
import json
data = json.load(open('dataset_hybrid_mesh_sp_er.json'))
print(f'✓ Instances: {len(data[\"instances\"])}')
types = {}
[types.update({i['topology_type'].split('_')[0]: types.get(i['topology_type'].split('_')[0], 0) + 1}) for i in data['instances']]
print(f'✓ Breakdown: {types}')
"
```

### **Valider le format PyTorch**
```bash
python validate_gnn_format.py
```

### **Convertir en PyTorch si needed**
```python
import json, torch
data = json.load(open('dataset_hybrid_mesh_sp_er.json'))
for inst in data['instances']:
    torch.save({
        'x': torch.tensor(inst['x']),
        'y': torch.tensor(inst['y'])
    }, f"graphs/graph_{inst['id']}.pt")
```

---

## 💡 Conseils

1. **Lancer la nuit** : 33h de calcul → non-bloquant
2. **CPU stable** : Utiliser 3-4 workers max (sinon throttling)
3. **Monitoring** : `watch -n 10 'tail -5 generation_progress.log'`
4. **Interruption sûre** : Ctrl+C arrête proprement la génération (JSON sauvegardé)

---

## 📞 En cas de problème

- **Erreur mémoire** : Réduire `n_workers` (ligne 183)
- **Graphes invalides** : Vérifier `generate_er.py` (pruning)
- **Timeout** : Augmenter `iters` dans configs des grosses graphes

---

**Status** : ✅ Code prêt pour 50 000 instances
