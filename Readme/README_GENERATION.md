# ✅ Système de Génération 50 000 Graphes - PRÊT À LANCER

## 📋 Résumé

Le système est **100% prêt** pour générer 50 000 graphes hybrides (MESH + SP + ER) en ~33 heures.

### Améliorations ajoutées :

✅ **Barre de progression** (tqdm) - suivi en temps réel  
✅ **Fichier log automatique** (`generation_progress.log`)  
✅ **Scripts de lancement** (`launch_generation.sh`)  
✅ **Monitoring en direct** (`monitor_generation.sh`)  
✅ **Statistiques complètes** après génération  

---

## 🚀 Comment utiliser

### **Méthode 1 : Lancement simple (terminal actif)**

```bash
cd /users/local/l24virbe/Projet_ML
source env_projet/bin/activate
python main_production.py
```

**Avantage** : Voir la barre de progression en temps réel  
**Inconvénient** : Doit garder le terminal ouvert

---

### **Méthode 2 : Lancement en arrière-plan (recommandé) 🌙**

```bash
cd /users/local/l24virbe/Projet_ML
./launch_generation.sh
```

**Avantage** : 
- Exécution non-bloquante
- Peut fermer le terminal
- Logs sauvegardés automatiquement
- Format avec couleurs

---

### **Méthode 3 : Lancement ultra-discret (SSH/serveur)**

```bash
nohup python /users/local/l24virbe/Projet_ML/main_production.py > /users/local/l24virbe/Projet_ML/generation.out 2>&1 &
```

Puis dans une autre session :
```bash
tail -f /users/local/l24virbe/Projet_ML/generation.out
```

---

## 📊 Suivre la progression

### **Pendant la génération**

#### Option A : Monitoring en direct
```bash
./monitor_generation.sh
```

Affiche toutes les 30s :
- ✓ Nombre d'instances générées
- ✓ Pourcentage de completion
- ✓ Débit (graphes/heure)
- ✓ Usage CPU/RAM
- ✓ ETA estimée

#### Option B : Consulter le log text
```bash
tail -f generation_progress.log
```

Ou regarder l'output brut :
```bash
tail -f generation_output.log
```

### **Après la génération**

Les résultats s'affichent automatiquement :
```
======================================================================
✅ GÉNÉRATION COMPLÈTE
======================================================================

📊 Instances : 50000
   MESH: 16667 | SP: 16667 | ER: 16666

⏱️  Temps total : 32h 58m 23.5s
   Temps moyen/graphe : 2.38s
   Débit : 1511 graphes/heure

📁 Fichier : dataset_hybrid_mesh_sp_er.json
   Taille : 250.8 MB

📈 Statistiques générées :
   • H (Horizon) : 5-20 (moy: 12.0)
   • B (Budget)  : 1.00-5.00 (moy: 2.99)
   • J* (Fiabilité) : 0.0000-1.0000 (moy: 0.64±0.38)
   • Nœuds/graphe : 3-12 (moy: 7.3)

======================================================================
```

---

## ⏱️ Estimation des temps (basée sur test 100 graphes)

| Instances | Durée estimée | Notes |
|-----------|---------------|-------|
| 10        | < 1 min       | Test rapide |
| 100       | 4 min         | ✅ Testé |
| 1 000     | 40 min        | Extrapolation |
| 10 000    | 6h 40 min     | Extrapolation |
| **50 000**| **~33 heures**| 3 workers, 2.4s/graphe |

⚠️ **Facteurs variables** :
- Vitesse CPU du serveur
- Charge système
- Taille des graphes (ER peut être lent)

---

## 📁 Fichiers générés

### **Principaux**
- `dataset_hybrid_mesh_sp_er.json` - Dataset complet (250 MB)
- `generation_progress.log` - Résumé avec stats

### **Optionnels (si lancement en arrière-plan)**
- `generation_output.log` - Output complet avec barre tqdm
- `generation.out` - Output brut (si nohup)

---

## 🔧 Paramètres ajustables

Si vous voulez modifier la génération, éditez [main_production.py](main_production.py) :

```python
# Ligne 138
n_instances = 50000  # Nombre d'instances

# Ligne 140-141
H_min, H_max = 5, 20        # Plage horizon
B_min, B_max = 1.0, 5.0     # Plage budget

# Ligne 183
n_workers = min(3, cpu_count(), n_instances)  # Modifier le 3 pour + workers
```

---

## ✅ Avant de lancer

- [x] `main_production.py` configuré avec `n_instances = 50000`
- [x] 3 générateurs intégrés (MESH, SP, ER)
- [x] tqdm installé pour barre de progression
- [x] Scripts de lancement prêts
- [x] Logs automatiques configurés

---

## 🎯 Checklist finale

```bash
# 1. Vérifier que tout est bon
cd /users/local/l24virbe/Projet_ML
python -c "from generate_mesh1 import *; from generate_sp1 import *; from generate_er import *; print('✓ Tous les générateurs OK')"

# 2. Vérifier tqdm
python -c "from tqdm import tqdm; print('✓ tqdm OK')"

# 3. Lancer la génération
./launch_generation.sh  # Ou python main_production.py
```

---

## 🆘 Troubleshooting

### **La génération s'arrête**
→ Vérifier le log : `tail -100 generation_output.log`

### **Trop lent**
→ Réduire `n_workers` (cause overload)
→ Vérifier charge système : `htop`

### **Erreur mémoire**
→ Réduire `n_workers` de 3 à 2

### **Besoin d'arrêter**
→ Ctrl+C (arrêt propre, JSON sauvegardé)

---

## 📞 Questions ?

Consultez le [GUIDE_GENERATION_50K.md](GUIDE_GENERATION_50K.md) pour plus de détails.

---

**Status** : 🟢 **PRÊT** - Vous pouvez lancer la génération !
