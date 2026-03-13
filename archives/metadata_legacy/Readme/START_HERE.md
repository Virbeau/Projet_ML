# Projet ML - Génération 50k Graphes

## 🚀 Lancer la génération (façon simple)

```bash
cd /users/local/l24virbe/Projet_ML
./launch_detached.sh
```

Puis fermez le terminal → **génération continue même si vous fermez votre VM!**

---

## 📊 Pendant la génération

Vérifier l'état :
```bash
./status_generation.sh
```

Voir en direct :
```bash
tail -f generation_output.log
```

---

## 🛑 Arrêter (si besoin)

```bash
./stop_generation.sh
```

---

## 📁 Fichiers disponibles

### Documentation
- **[CHECKLIST_FINAL.md](CHECKLIST_FINAL.md)** ← Lire avant de lancer!
- **[GUIDE_MACHINE_VIRTUELLE.md](GUIDE_MACHINE_VIRTUELLE.md)** - Guide complet pour VM
- **[README_GENERATION.md](README_GENERATION.md)** - Détails techniques
- **[GUIDE_GENERATION_50K.md](GUIDE_GENERATION_50K.md)** - Infos détaillées

### Scripts prêts à utiliser
- `launch_detached.sh` - Lancer (persistant)
- `status_generation.sh` - Voir l'état
- `stop_generation.sh` - Arrêter proprement
- `monitor_generation.sh` - Dashboard live

### Code principal
- `main_production.py` - Générateur principal (50k configuré)
- `generate_mesh1.py` - Graphes Mesh
- `generate_sp1.py` - Graphes Série-Parallèle
- `generate_er.py` - Graphes Erdős-Rényi
- `solver.py` - Solveur pour allocation budgets

---

## ✅ Quick Start

```bash
# 1. Lancer
./launch_detached.sh

# 2. Fermer terminal (génération continue)

# 3. Plus tard, vérifier
./status_generation.sh
```

---

## 📊 Détails

- **Instances** : 50 000 graphes
- **Répartition** : 1/3 MESH + 1/3 SP + 1/3 ER
- **Temps estimé** : ~33 heures
- **Débit** : ~1500 graphes/heure
- **Output** : `dataset_hybrid_mesh_sp_er.json` (250 MB)
- **Format** : PyTorch Geometric compatible

---

## 🤔 Besoin d'aide?

Lire [CHECKLIST_FINAL.md](CHECKLIST_FINAL.md) pour tout détail.

