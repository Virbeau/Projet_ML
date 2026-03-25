# Restitution – Projet GNN et Génération de Datasets (Branche restitution)

Ce dépôt contient l’ensemble du code, des scripts, des jeux de données et des éléments nécessaires à la restitution du projet auprès des enseignants. Cette branche a été spécialement préparée pour garantir la clarté, la reproductibilité et la neutralité des commentaires dans tous les scripts.

## Contenu principal de la branche

- **Scripts de génération de datasets** :
  - `generate_mesh1.py`, `generate_er.py`, `generate_sp1.py`, `generate_v7_family_sets.py`, `fusion_datasetV7.py`
- **Scripts d’entraînement et de prédiction GNN** :
  - `GINE_B_repartition.py`, `GINE_J_predictor.py`, `GraphSAGE_B_repartition.py`, `GraphSAGE_J_predictor.py`
- **Solver et validation Monte Carlo** :
  - `solver.py`, `monte_carlo_first_instances.py`, `monte_carlo_validation.py`
- **Notebooks et visualisations** :
  - `archives/notebooks/Notebook_Explication_Generation_Dataset_Autonome.ipynb` (explications détaillées)
  - `visu_b_allocation_gine.py` (script de visualisation)
  - `allocation_comparison_GINE_B.png` (image de comparaison)
- **Jeux de données** :
  - Archive `datasetsV7.zip` contenant tous les fichiers JSON de datasets V7 (mesh, er, sp, fusion, testsets)

## Structure recommandée pour la restitution

1. **Génération des datasets** :
   - Utiliser les scripts de génération pour produire les différents jeux de données (mesh, er, sp, etc.).
   - Fusionner les datasets si besoin avec `fusion_datasetV7.py`.
2. **Entraînement et évaluation des modèles GNN** :
   - Scripts pour GINE et GraphSAGE, pour les tâches B et J.
   - Les checkpoints sont à placer dans le dossier `checkpoints/`.
3. **Validation et analyse** :
   - Utiliser les scripts Monte Carlo pour valider la robustesse et la disponibilité sur les instances générées.
   - Visualiser les résultats avec les scripts et images fournis.
4. **Notebooks** :
   - Le notebook fourni explique la démarche de génération et d’utilisation des datasets.

## Remarques importantes
- Tous les scripts ont été commentés de façon neutre et explicative, sans référence à des versions antérieures ou à des choix personnels.
- Les jeux de données sont fournis sous forme d’archive pour faciliter la distribution.
- Cette branche ne contient que les éléments strictement nécessaires à la restitution.

## Contact
Pour toute question ou précision, merci de contacter l’auteur du projet.
