# ✅ CHECKLIST FINALE - 50K GENERATION

**Status** : 🟢 **PRÊT À LANCER**

---

## 🎯 Objectif final

Générer **50 000 graphes hybrides** (1/3 MESH + 1/3 SP + 1/3 ER) en ~33 heures sur une VM, avec suivi en temps réel et persistance après fermeture.

---

## ✅ Vérifications pré-lancement

### Code & Configuration
- [x] `main_production.py` : `n_instances = 50000` (ligne 138)
- [x] 3 générateurs intégrés : MESH, SP, ER
- [x] Conversion GNN-ready (x, y matrices)
- [x] Métadonnées complètes dans JSON

### Dépendances
- [x] NetworkX 3.6.1
- [x] NumPy 2.4.2
- [x] tqdm (installé pour barre progress)

### Scripts de lancement
- [x] `launch_detached.sh` - Lancement détaché (screen/tmux/nohup)
- [x] `status_generation.sh` - Vérifier l'état
- [x] `stop_generation.sh` - Arrêt gracieux
- [x] `monitor_generation.sh` - Dashboard en direct

### Documentation
- [x] `GUIDE_MACHINE_VIRTUELLE.md` - Pour utilisation VM
- [x] `README_GENERATION.md` - Guide complet
- [x] `GUIDE_GENERATION_50K.md` - Détails techniques

---

## 🚀 Lancement (en 3 étapes)

### **Étape 1 : Ouvrir terminal sur votre VM**

```bash
ssh user@vm         # Si distant
cd /users/local/l24virbe/Projet_ML
```

### **Étape 2 : Lancer avec persistance**

```bash
./launch_detached.sh
```

**Vous verrez :**
```
⚙️  Génération en cours (3 workers)...
📊 Suivi disponible dans : generation_progress.log

✓ Session détachée créée
  Session : generation_50k
  
📋 Pour reprendre la session :
  screen -r generation_50k

📋 Pour voir en direct :
  tail -f generation_output.log
```

### **Étape 3 : Fermer le terminal (génération continue!)**

```bash
exit
```

**La génération continue même si vous fermez tout.** ✅

---

## 📊 Suivi de la génération

### Pendant l'exécution (en direct)

```bash
# Option 1 : Voir les logs
tail -f generation_output.log

# Option 2 : Dashboard live (refresh 30s)
./monitor_generation.sh

# Option 3 : Reprendre la session détachée
screen -r generation_50k
```

### Plus tard (vérifier l'état)

```bash
# Revenir plus tard
ssh user@vm
cd /users/local/l24virbe/Projet_ML

# Vérifier la progression
./status_generation.sh
```

**Affichera** :
- État de la session
- Nombre d'instances générées
- Taille du dataset partiel
- Logs récents

---

## 📁 Fichiers produits

### Principal
- **`dataset_hybrid_mesh_sp_er.json`** (~250 MB)
  - 50 000 instances
  - Format GNN-ready (x, y matrices)
  - Métadonnées complètes

### Logs
- **`generation_progress.log`** - Résumé avec stats
- **`generation_output.log`** - Output brut avec barre tqdm
- **`generation.pid`** - PID file (auto-créé)

---

## ⏱️ Timing estimé

| Phase | Durée |
|-------|-------|
| Configuration | 1 min |
| Lancement | 2 min |
| Génération réelle | **~33 heures** |
| Arrêt + cleanup | 1 min |
| **Total** | **~33 heures** |

**Débit** : ~1500 graphes/heure

---

## 🎮 Contrôle du processus

### Voir l'état en direct
```bash
./status_generation.sh
```

### Arrêter proprement
```bash
./stop_generation.sh
```
(Arrêt gracieux + SIGKILL si besoin + cleanup)

### Reprendre la session (if using screen)
```bash
screen -r generation_50k
```

---

## 🔍 Validation post-génération

Une fois terminée, vérifier l'intégrité :

```bash
# Vérifier le JSON
python -c "
import json
data = json.load(open('dataset_hybrid_mesh_sp_er.json'))
print(f'✓ Instances: {len(data[\"instances\"])}')
types = {}
[types.update({i['topology_type'].split('_')[0]: types.get(i['topology_type'].split('_')[0], 0) + 1}) for i in data['instances']]
print(f'✓ Breakdown: {types}')
"

# Valider le format PyTorch Geometric
python validate_gnn_format.py
```

---

## 💡 Tips & Tricks

### Si vous avez peu de temps
- Lancer avec `./launch_detached.sh`
- Fermer terminal immédiatement
- La génération continue tout seul

### Si vous voulez monitorer
- Utiliser `tail -f generation_output.log`
- Ou `./monitor_generation.sh` pour dashboard
- Ou reprendre la session `screen -r generation_50k`

### Si la VM crash
- Dataset est sauvegardé progressivement (JSON valide)
- Relancer `./launch_detached.sh` pour continuer (génèrera des duplicatas cependant)
- Pour éviter duplicatas : modifier le code pour skipper instances existantes

### Si vous voulez arrêter avant la fin
```bash
./stop_generation.sh
```
Les données générées jusqu'à présent restent dans le JSON.

---

## 🚨 Cas problématiques

### La génération s'arrête sans raison
→ Vérifier : `tail -50 generation_output.log`

### Trop lent ?
→ Réduire `n_workers` de 3 à 2 dans `main_production.py` (line 183)

### Erreur mémoire ?
→ Réduire `n_workers` ou augmenter RAM allouée à la VM

### Besoin d'accélérer plus ?
→ Augmenter `n_workers` à 4 (si CPU disponible)

---

## 📞 Questions fréquentes

**Q: Mon terminal se ferme, ça tue la génération?**  
R: Non! Utilisez `./launch_detached.sh` → génération persiste

**Q: Puis-je vérifier l'avancement?**  
R: Oui! `./status_generation.sh` + `tail -f generation_output.log`

**Q: Combien de temps pour 50k?**  
R: ~33 heures (1500 graphes/h)

**Q: Je peux éteindre ma machine?**  
R: Oui! La VM continue. Juste la VM, pas votre PC hôte.

**Q: Puis-je relancer si interrompue?**  
R: Oui, mais générera des duplicatas. Meilleur = modifier le code pour skipper instances existantes.

---

## ✅ Final Checklist

Avant de lancer :
- [ ] Terminal ouvert sur la VM
- [ ] Dans le bon répertoire : `/users/local/l24virbe/Projet_ML`
- [ ] Lu le guide : `GUIDE_MACHINE_VIRTUELLE.md`
- [ ] Vérifié que `main_production.py` a `n_instances = 50000`

Avant de fermer :
- [ ] Lancé `./launch_detached.sh`
- [ ] Noté le message (session/PID)
- [ ] Fermé le terminal → génération continue ✓

Pour plus tard :
- [ ] Utilisé `./status_generation.sh` pour vérifier
- [ ] Utilisé `./monitor_generation.sh` pour dashboard
- [ ] Utilisé `tail -f generation_output.log` pour logs en direct

---

## 🎯 RÉSUMÉ ULTRA-COURT

```bash
# Sur la VM
./launch_detached.sh

# Fermer terminal → génération continue!

# Plus tard pour vérifier
./status_generation.sh
```

**C'est tout ce que vous avez besoin de faire.** ✅

---

**Status Final** : 🟢 **PRÊT À LANCER**

Vous pouvez maintenant lancer la génération 50k sans crainte! 🚀

