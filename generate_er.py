# Importation des bibliothèques nécessaires pour la manipulation de graphes, le calcul numérique et l'aléatoire
import networkx as nx
import numpy as np
import random


def prune_useless_nodes(G, source, target):
    """
    Retourne un sous-graphe contenant uniquement les nœuds qui relient la source à la cible.
    """
    try:
        reachable_from_source = set(nx.descendants(G, source)) | {source}
        can_reach_target = set(nx.ancestors(G, target)) | {target}
        useful_nodes = reachable_from_source.intersection(can_reach_target)
        return G.subgraph(useful_nodes).copy()
    except Exception:
        return G


def generate_er_instance(num_nodes=10, p=0.3, seed=None):
    """
    Génère un graphe orienté de type Erdős-Rényi et retourne le graphe ainsi qu'un dictionnaire descriptif.
    """
    # Initialisation des générateurs aléatoires pour garantir la reproductibilité si un seed est fourni
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    while True:
        # Génération d'un graphe orienté aléatoire avec le nombre de nœuds et la probabilité d'arête spécifiés
        G_raw = nx.erdos_renyi_graph(n=num_nodes, p=p, directed=True)
        nodes = list(G_raw.nodes())

        # Vérification d'un nombre minimal de nœuds pour garantir un problème intéressant
        if len(nodes) < 3:
            continue

        # Sélection de deux nœuds distincts pour la source et la cible, distants d'au moins 2 sauts
        # Ceci évite le cas trivial où la source et la cible sont directement connectées
        source_node, target_node = random.sample(nodes, 2)
        if nx.has_path(G_raw, source_node, target_node):
            dist = nx.shortest_path_length(G_raw, source_node, target_node)
            if dist >= 2:
                # Suppression des nœuds inutiles (non connectés entre source et cible)
                G = prune_useless_nodes(G_raw, source_node, target_node)
                # Vérification de la taille minimale après nettoyage
                if len(G.nodes()) >= 3:
                    break

    # Définition des nœuds terminaux (source et cible)
    terminals = [source_node, target_node]
    # Liste des nœuds réparables (tous sauf les terminaux)
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
            p_fail = round(random.uniform(0.12, 0.24), 3)
            c_cost = round(random.uniform(1.0, 10.0), 2)

        # Calcul de la distance minimale jusqu'à la cible
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
    # Construction du dictionnaire décrivant l'instance générée
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
        # Extraction des probabilités de panne et coûts sous forme de listes
        "p_fail_array": [node_features[v]["p_fail"] for v in G.nodes()],
        "c_cost_array": [node_features[v]["c_cost"] for v in G.nodes()]
    }

    # Retourne le graphe et la description de l'instance
    return G, instance_record