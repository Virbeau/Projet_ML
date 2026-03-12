import networkx as nx
import numpy as np
import random

def generate_sp_instance(num_repairable=10, seed=None):
    """
    Génère un graphe Série-Parallèle en insérant itérativement des noeuds.
    num_repairable: Le nombre de noeuds intermédiaires (routeurs/machines).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    # 1. Initialisation avec le graphe le plus simple : Source -> Cible
    G = nx.DiGraph()
    source_node = 0
    target_node = 1
    G.add_edge(source_node, target_node)
    
    # 2. Construction procédurale ALÉATOIRE (Série / Parallèle + Boucles)
    current_node_id = 2
    while current_node_id < num_repairable + 2: # +2 pour compter la Source et la Cible
        edges = list(G.edges())
        u, v = random.choice(edges) # On choisit une arête au hasard
        
        # Choix aléatoire de l'opération (40% série, 40% parallèle, 20% boucle)
        operation = random.random()
        
        if operation < 0.4:
            # OPÉRATION SÉRIE : On coupe l'arête et on insère le noeud au milieu
            G.remove_edge(u, v)
            G.add_edge(u, current_node_id)
            G.add_edge(current_node_id, v)
            
        elif operation < 0.8:
            # OPÉRATION PARALLÈLE : On crée une voie de contournement autour de l'arête
            G.add_edge(u, current_node_id)
            G.add_edge(current_node_id, v)
            
        else:
            # OPÉRATION BOUCLE : On crée un cycle en connectant à un noeud existant
            # On ajoute le nouveau noeud entre u et v, puis on crée une boucle
            G.remove_edge(u, v)
            G.add_edge(u, current_node_id)
            G.add_edge(current_node_id, v)
            
            # On ajoute une arête vers un noeud aléatoire accessible depuis current_node_id
            # pour créer des chemins alternatifs
            potential_targets = [n for n in G.nodes() if n != current_node_id and n != u]
            if potential_targets and random.random() < 0.5:
                loop_target = random.choice(potential_targets)
                # On ajoute soit une arête entrante, soit sortante aléatoirement
                if random.random() < 0.5:
                    G.add_edge(current_node_id, loop_target)
                else:
                    G.add_edge(loop_target, current_node_id)
            
        current_node_id += 1

    # 3. Préparation des listes pour le contrat de données
    terminals = [source_node, target_node]
    repairable_nodes = [v for v in G.nodes() if v not in terminals]
    
    # 4. Calcul des 7 Features (Strictement identique à generate_mesh1.py)
    node_features = {}
    for v in G.nodes():
        is_source = 1 if v == source_node else 0
        is_target = 1 if v == target_node else 0
        
        if v in terminals:
            p_fail, c_cost = 0.0, 0.0
        else:
            p_fail = round(random.uniform(0.05, 0.10), 3)
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