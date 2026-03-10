# 🖥️ Lancement pour Machine Virtuelle - GUIDE COMPLET

## 🎯 Votre situation

Vous utilisez une **machine virtuelle** et voulez lancer la génération **50k** sans perdre le processus quand vous fermerez votre ordinateur.

**Solution** : Scripts de lancement **détachés** qui persistent après fermeture SSH/terminal.

---

## ✅ Prérequis vérifiés

- ✅ `main_production.py` configuré (50 000 instances)
- ✅ 3 générateurs intégrés (MESH, SP, ER)
- ✅ `tqdm` installé
- ✅ Scripts de détachement prêts

---

## 🚀 Lancer (façon simple et sûre)

### **1️⃣ Depuis votre VM, ouvrir terminal et faire :**

```bash
cd /users/local/l24virbe/Projet_ML
./launch_detached.sh
```

### **2️⃣ Le script va :**
- ✓ Détecter si `screen` ou `tmux` disponible
- ✓ Lancer la génération **détachée** (non-bloquante)
- ✓ Vous donner des instructions pour reprendre

**Output typique** :
```
⚙️  Génération en cours (3 workers)...
📊 Suivi disponible dans : generation_progress.log

✓ Session détachée créée
  Session : generation_50k
  PID : 12345

📋 Pour reprendre la session :
  screen -r generation_50k

📋 Pour voir en direct :
  tail -f generation_output.log
```

### **3️⃣ Maintenant vous pouvez :**
- ✅ Fermer votre terminal → **génération continue**
- ✅ Fermer votre SSH → **génération continue**
- ✅ Éteindre votre ordinateur → **génération continue** (sur la VM)
- ✅ Revenir plus tard et reprendre

---

## 📊 Suivi de l'état (quand vous reviennez)

### **Vérifier le statut depuis n'importe quel terminal :**

```bash
cd /users/local/l24virbe/Projet_ML
./status_generation.sh
```

**Affiche** :
- Session actuelle (screen/tmux)
- PID du processus
- Dernier output
- Taille du dataset généré
- Commandes disponibles

### **Voir la progression en direct :**

```bash
# Option 1 : Reprendre la session
screen -r generation_50k    # Si utilisé screen
tmux attach-session -t generation_50k  # Si utilisé tmux

# Option 2 : Voir les logs
tail -f generation_output.log
tail -f generation_progress.log
```

---

## 🛑 Arrêter proprement (si besoin)

```bash
./stop_generation.sh
```

Ça va :
- ✓ Arrêter la session (graceful SIGTERM + SIGKILL si nécessaire)
- ✓ Sauvegarder le dataset généré jusqu'à présent
- ✓ Garder les logs

**Note** : Même si interrompue, les données sont sauvegardées (JSON valide).

---

## 📋 Cycle complet exemple

### **Jour 1 - Lancer (soir)**
```bash
cd /users/local/l24virbe/Projet_ML
./launch_detached.sh
# Résultat: Session détachée created
# Vous fermez tout → génération continue
```

### **Jour 2 - Vérifier (matin)**
```bash
# Vous reconnectez à votre VM
cd /users/local/l24virbe/Projet_ML
./status_generation.sh
# Affiche: Génération à 45%, 18k instances, ~16h estimé
```

### **Jour 2 - Reprendre la session si vous voulez (optionnel)**
```bash
screen -r generation_50k
# ou
tail -f generation_output.log
```

### **Jour 2.5 - Terminée**
```bash
./status_generation.sh
# Affiche: Terminée, 50k instances, ~250 MB
# Le fichier dataset_hybrid_mesh_sp_er.json est ready
```

---

## 🔧 Fonctionnement technique

### **Qu'est-ce qui permet la persistance?**

**Option 1 : screen** (meilleur pour VM)
```bash
screen -dmS generation_50k ...  # Détaché, persiste
```
- Session persiste même sans tty
- Peut reprendre à distance
- Survit à SSH disconnect

**Option 2 : tmux** (alternative)
```bash
tmux new-session -d -s generation_50k ...
```
- Même principe que screen
- Plus moderne, meilleur split/resize

**Option 3 : nohup** (fallback)
```bash
nohup python main_production.py > output.log 2>&1 &
```
- Ignore SIGHUP (fermeture SSH)
- Laisse processus orphelin
- Moins de contrôle

---

## 💡 Cas particuliers

### **Je veux lancer depuis ma machine Windows/Mac et VM continue**

**Option A** : Utilisez SSH persistent
```bash
# Sur votre machine
nice -n 19 ssh user@vm './launch_detached.sh'
# Vous pouvez fermer le terminal → continue sur VM
```

**Option B** : Utilisez screen sur VM, reconnectez plus tard
```bash
# Jour 1
ssh user@vm
./launch_detached.sh
exit  # Ferme SSH, génération continue

# Jour 2
ssh user@vm
./status_generation.sh  # Voir l'état
screen -r generation_50k  # Reprendre si vous voulez
```

### **La VM s'éteint/crash pendant la génération?**

Les données sont **sauvegardées progressivement** dans JSON. Vous pouvez reprendre plus tard (mais il refaudra relancer entièrement).

Pour une vraie reprise (éviter calculs dupliqués), il faudrait modifier le code pour skipper les instances déjà générées.

---

## 📞 Commandes rapides

```bash
# Lancer détaché
./launch_detached.sh

# Voir l'état
./status_generation.sh

# Voir en temps réel (une de ces options)
tail -f generation_output.log          # Sortie brute
tail -f generation_progress.log        # Log de progression
screen -r generation_50k               # Reprendre session
monitor_generation.sh                  # Dashboard live

# Arrêter proprement
./stop_generation.sh
```

---

## ✅ Checklist avant de partir

- [ ] Lancé avec `./launch_detached.sh`
- [ ] Noté la session (screen/tmux) ou PID
- [ ] Fermer terminal → génération continue ✓
- [ ] Revenir demain pour voir l'état

---

## 🎯 RÉSUMÉ ULTRA-COURT

```bash
# Lancer (et fermer terminal)
./launch_detached.sh

# Plus tard, vérifier l'état
./status_generation.sh

# Reprendre si besoin
screen -r generation_50k
```

**C'est tout !** La VM fera le travail pendant que vous êtes offline. ✅

