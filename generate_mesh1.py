
# Importation des bibliothèques nécessaires pour la manipulation de graphes, le calcul numérique, l'aléatoire et la gestion JSON
import networkx as nx
import numpy as np
import random
import json


def generate_mesh_instance(m=3, n=4, seed=None):
    """
    Génère un graphe de type mesh (grille) orienté, avec attribution de caractéristiques à chaque nœud.
    Retourne le graphe NetworkX et un dictionnaire décrivant l'instance.
    """
    if seed is not None:
        # Initialisation des générateurs aléatoires pour reproductibilité
        random.seed(seed)
        np.random.seed(seed)

    # Création d'une grille 2D de taille m x n
    G_grid = nx.grid_2d_graph(m, n)
    # Conversion en graphe orienté pour permettre le calcul des degrés entrants/sortants
    G = G_grid.to_directed()

    # Remplacement des identifiants de nœuds (tuple) par des entiers consécutifs
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # Définition des nœuds terminaux (source et cible) dans les coins opposés de la grille
    source_node = mapping[(0, 0)]
    target_node = mapping[(m-1, n-1)]

    terminals = [source_node, target_node]
    # Liste des nœuds réparables (tous sauf les terminaux)
    repairable_nodes = [v for v in G.nodes() if v not in terminals]

    # Calcul des caractéristiques pour chaque nœud
    node_features = {}
    for v in G.nodes():
        # Indicateurs pour la source et la cible
        is_source = 1 if v == source_node else 0
        is_target = 1 if v == target_node else 0

        # Attribution des probabilités de panne et coûts
        if v in terminals:
            p_fail = 0.0
            c_cost = 0.0
        else:
            p_fail = round(random.uniform(0.16, 0.32), 3)
            c_cost = round(random.uniform(1.0, 10.0), 2)

        # Calcul des degrés entrants et sortants
        in_degree = G.in_degree(v)
        out_degree = G.out_degree(v)

        # Calcul de la distance minimale jusqu'à la cible
        try:
            dist_to_target = nx.shortest_path_length(G, source=v, target=target_node)
        except nx.NetworkXNoPath:
            dist_to_target = 999  # Valeur élevée si le nœud est isolé

        node_features[v] = {
            "p_fail": p_fail,
            "c_cost": c_cost,
            "is_source": is_source,
            "is_target": is_target,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "distance_to_target": dist_to_target
        }

    # Construction du dictionnaire décrivant l'instance
    instance_record = {
        "topology": f"mesh_grid_{m}x{n}",
        "graph": {
            "nodes": list(G.nodes()),
            "edges": list(G.edges()),
            "is_directed": True
        },
        "terminals": terminals,
        "repairable_nodes": repairable_nodes,
        "features": node_features,
        # Extraction des probabilités de panne et coûts sous forme de listes
        "p_fail_array": [node_features[v]["p_fail"] for v in G.nodes()],
        "c_cost_array": [node_features[v]["c_cost"] for v in G.nodes()]
    }

    return G, instance_record

# Exemple d'utilisation du script en mode autonome
if __name__ == "__main__":
    # Génération d'une instance de mesh pour test
    G_test, record_test = generate_mesh_instance(m=3, n=4, seed=42)

    # Affichage des informations principales de l'instance générée
    print("=== APERÇU DE L'INSTANCE GÉNÉRÉE ===")
    print(f"Topologie : {record_test['topology']}")
    print(f"Nombre de noeuds total : {len(G_test.nodes())}")
    print(f"Nombre de noeuds réparables : {len(record_test['repairable_nodes'])}")
    print("\n--- Caractéristiques du noeud source (0) ---")
    print(json.dumps(record_test['features'][0], indent=2))
    print("\n--- Caractéristiques d'un noeud réparateur (5) ---")
    print(json.dumps(record_test['features'][5], indent=2))