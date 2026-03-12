import networkx as nx
import numpy as np
import random

def prune_useless_nodes(G, source, target):
    """Garde uniquement les noeuds utiles qui relient S à T"""
    try:
        reachable_from_source = set(nx.descendants(G, source)) | {source}
        can_reach_target = set(nx.ancestors(G, target)) | {target}
        useful_nodes = reachable_from_source.intersection(can_reach_target)
        return G.subgraph(useful_nodes).copy()
    except Exception:
        return G

def generate_er_instance(num_nodes=10, p=0.3, seed=None):
    """
    Génère un graphe d'Erdős-Rényi orienté.
    Peut générer de très petits graphes (ex: 5 noeuds) comme des plus grands (ex: 15 noeuds).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    while True:
        # 1. On génère le graphe brut avec la taille demandée
        G_raw = nx.erdos_renyi_graph(n=num_nodes, p=p, directed=True)
        nodes = list(G_raw.nodes())
        
        if len(nodes) < 3:
            continue
            
        # 2. On choisit S et T en s'assurant qu'ils sont distants d'au moins 2 sauts
        # (Pour éviter le chemin trivial S -> T direct sans noeuds intermédiaires)
        source_node, target_node = random.sample(nodes, 2)
        if nx.has_path(G_raw, source_node, target_node):
            dist = nx.shortest_path_length(G_raw, source_node, target_node)
            if dist >= 2:
                # 3. Nettoyage des noeuds qui ne servent à rien
                G = prune_useless_nodes(G_raw, source_node, target_node)
                
                # 4. Vérification de la taille finale
                # Le minimum vital pour un problème intéressant est 3 noeuds (S, T, et 1 routeur)
                if len(G.nodes()) >= 3: 
                    break # Graphe valide trouvé !

    terminals = [source_node, target_node]
    repairable_nodes = [v for v in G.nodes() if v not in terminals]
    
    # 5. Calcul des features (Le Contrat de Données)
    node_features = {}
    for v in G.nodes():
        is_source = 1 if v == source_node else 0
        is_target = 1 if v == target_node else 0
        
        if v in terminals:
            p_fail, c_cost = 0.0, 0.0
        else:
            p_fail = round(random.uniform(0.05, 0.10), 3)
            c_cost = round(random.uniform(1.0, 10.0), 2)
            
        try:
            dist_to_target = nx.shortest_path_length(G, source=v, target=target_node)
        except nx.NetworkXNoPath:
            dist_to_target = 999 
            
        node_features[v] = {
            "p_fail": p_fail,
            "c_cost": c_cost,
            "is_source": is_source,
            "is_target": is_target,
            "in_degree": G.in_degree(v),
            "out_degree": G.out_degree(v),
            "distance_to_target": dist_to_target
        }
        
    instance_record = {
        "topology": f"erdos_renyi_{len(G.nodes())}_nodes_p{p}",
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