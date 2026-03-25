
# Importation des bibliothèques pour la manipulation de graphes, le calcul numérique et l'aléatoire
import networkx as nx
import numpy as np
import random

def generate_sp_instance(num_repairable=10, seed=None):
    """
    Génère un graphe orienté de type Série-Parallèle en insérant itérativement des nœuds intermédiaires.
    num_repairable : nombre de nœuds intermédiaires (hors source et cible).
    """
    # Initialisation des générateurs aléatoires pour la reproductibilité
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Création du graphe initial avec une arête de la source à la cible
    G = nx.DiGraph()
    source_node = 0
    target_node = 1
    G.add_edge(source_node, target_node)

    # Construction procédurale aléatoire : insertion de nœuds en série, parallèle ou boucle
    current_node_id = 2
    while current_node_id < num_repairable + 2:  # +2 pour inclure source et cible
        edges = list(G.edges())
        u, v = random.choice(edges)  # Sélection aléatoire d'une arête

        # Choix aléatoire de l'opération à effectuer
        operation = random.random()

        if operation < 0.4:
            # Insertion en série : coupe l'arête et insère le nœud au milieu
            G.remove_edge(u, v)
            G.add_edge(u, current_node_id)
            G.add_edge(current_node_id, v)

        elif operation < 0.8:
            # Insertion en parallèle : ajoute un chemin alternatif autour de l'arête
            G.add_edge(u, current_node_id)
            G.add_edge(current_node_id, v)

        else:
            # Insertion avec boucle : crée un cycle en connectant à un nœud existant
            G.remove_edge(u, v)
            G.add_edge(u, current_node_id)
            G.add_edge(current_node_id, v)

            # Ajout d'une arête supplémentaire pour créer des chemins alternatifs
            potential_targets = [n for n in G.nodes() if n != current_node_id and n != u]
            if potential_targets and random.random() < 0.5:
                loop_target = random.choice(potential_targets)
                # Ajout d'une arête entrante ou sortante de façon aléatoire
                if random.random() < 0.5:
                    G.add_edge(current_node_id, loop_target)
                else:
                    G.add_edge(loop_target, current_node_id)

        current_node_id += 1

    # Préparation des listes de nœuds terminaux et réparables
    terminals = [source_node, target_node]
    repairable_nodes = [v for v in G.nodes() if v not in terminals]

    # Calcul des caractéristiques pour chaque nœud
    node_features = {}
    for v in G.nodes():
        is_source = 1 if v == source_node else 0
        is_target = 1 if v == target_node else 0

        # Attribution des probabilités de panne et coûts
        if v in terminals:
            p_fail, c_cost = 0.0, 0.0
        else:
            p_fail = round(random.uniform(0.10, 0.22), 3)
            c_cost = round(random.uniform(1.0, 10.0), 2)

        in_degree = G.in_degree(v)
        out_degree = G.out_degree(v)

        try:
            dist_to_target = nx.shortest_path_length(G, source=v, target=target_node)
        except nx.NetworkXNoPath:
            dist_to_target = 999 
            
        node_features[v] = {
            "p_fail": p_fail,
            "c_cost": c_cost,
            "is_source": is_source,
            "is_target": is_target,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "distance_to_target": dist_to_target
        }
        
    # 5. Formatage du Dictionnaire final
    instance_record = {
        "topology": f"series_parallel_{num_repairable}_nodes",
        "graph": {
            "nodes": list(G.nodes()),
            "edges": list(G.edges()),
            "is_directed": True
        },
        "terminals": terminals,
        "repairable_nodes": repairable_nodes,
        "features": node_features,
        "p_fail_array": [node_features[v]["p_fail"] for v in G.nodes()],
        "c_cost_array": [node_features[v]["c_cost"] for v in G.nodes()]
    }
    
    return G, instance_record