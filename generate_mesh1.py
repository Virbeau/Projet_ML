import networkx as nx
import numpy as np
import random
import json

def generate_mesh_instance(m=3, n=4, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    # 1. Génération de la grille (Mesh/Usine)
    # nx.grid_2d_graph crée une grille avec des coordonnées (x,y)
    G_grid = nx.grid_2d_graph(m, n)
    
    # On le convertit en graphe orienté (bi-directionnel) pour avoir des in/out degrees
    G = G_grid.to_directed() 
    
    # On renomme les noeuds (0,0), (0,1)... en entiers simples 0, 1, 2... pour le GNN
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # 2. Définition des Terminaux ALÉATOIRES (source et target à des positions différentes)
    all_nodes = list(G.nodes())
    # On choisit 2 noeuds au hasard pour source et target
    selected = random.sample(all_nodes, 2)
    source_node = selected[0]
    target_node = selected[1]
    
    # Vérification qu'il existe bien un chemin entre source et target
    # (dans une grille connexe, c'est toujours vrai mais on vérifie par sécurité)
    if not nx.has_path(G, source_node, target_node):
        # Si pas de chemin, on prend les coins opposés par défaut
        reverse_mapping = {v: k for k, v in mapping.items()}
        source_node = mapping[(0, 0)]
        target_node = mapping[(m-1, n-1)]
    
    terminals = [source_node, target_node]
    repairable_nodes = [v for v in G.nodes() if v not in terminals]
    
    # 3. Calcul des 7 Features pour chaque noeud
    node_features = {}
    for v in G.nodes():
        # Features 3 & 4 : Source / Target
        is_source = 1 if v == source_node else 0
        is_target = 1 if v == target_node else 0
        
        # Features 1 & 2 : p_fail et c_cost
        if v in terminals:
            # Les terminaux sont parfaits (pas de panne, pas de coût)
            p_fail = 0.0 
            c_cost = 0.0
        else:
            # Noeuds réparables : valeurs aléatoires réalistes
            p_fail = round(random.uniform(0.01, 0.40), 3) # 5% à 40% de chance de panne
            c_cost = round(random.uniform(1.0, 10.0), 2)   # Coût de réparation entre 1 et 10
            
        # Features 5 & 6 : Degrés (Combien de tuyaux arrivent et partent ?)
        in_degree = G.in_degree(v)
        out_degree = G.out_degree(v)
        
        # Feature 7 : Distance à la Cible (La feature la plus importante)
        try:
            dist_to_target = nx.shortest_path_length(G, source=v, target=target_node)
        except nx.NetworkXNoPath:
            dist_to_target = 999 # Sécurité si un noeud est totalement déconnecté
            
        node_features[v] = {
            "p_fail": p_fail,
            "c_cost": c_cost,
            "is_source": is_source,
            "is_target": is_target,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "distance_to_target": dist_to_target
        }
        
    # 4. Formatage du Dictionnaire final (Prêt pour le Solveur et le JSON)
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
        
        # Extractions sous forme de listes pour faciliter la vie du solveur du Notebook :
        "p_fail_array": [node_features[v]["p_fail"] for v in G.nodes()],
        "c_cost_array": [node_features[v]["c_cost"] for v in G.nodes()]
    }
    
    return G, instance_record

# --- TEST DU SCRIPT ---
if __name__ == "__main__":
    # Génère une instance de test
    G_test, record_test = generate_mesh_instance(m=3, n=4, seed=42)
    
    # Affiche le résultat joliment dans le terminal
    print("=== APERÇU DE L'INSTANCE GÉNÉRÉE ===")
    print(f"Topologie : {record_test['topology']}")
    print(f"Nombre de noeuds total : {len(G_test.nodes())}")
    print(f"Nombre de noeuds réparables : {len(record_test['repairable_nodes'])}")
    print("\n--- Features du noeud Source (0) ---")
    print(json.dumps(record_test['features'][0], indent=2))
    print("\n--- Features d'un noeud routeur/réparable (5) ---")
    print(json.dumps(record_test['features'][5], indent=2))