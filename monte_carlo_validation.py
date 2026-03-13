import json
import random
import numpy as np
import networkx as nx
from tqdm import tqdm

def fast_is_connected(G_base, source, target, up_nodes):
    """
    Vérifie la connectivité sur le sous-graphe des noeuds fonctionnels.
    """
    # Les terminaux sont toujours UP
    valid_nodes = set(up_nodes) | {source, target}
    subG = G_base.subgraph(valid_nodes)
    try:
        return nx.has_path(subG, source, target)
    except nx.NetworkXPointlessConcept:
        return False

def simulate_monte_carlo(instance, n_sims=10000):
    """
    Simule n_sims trajectoires markoviennes vectorisées pour un graphe donné.
    Calcule la Disponibilité (Availability) : Le pourcentage moyen de temps passé en panne.
    """
    H = int(instance["H"])
    edges = instance["graph"]["edges"]
    nodes = instance["graph"]["nodes"]  # On récupère la vraie liste des noeuds
    terminals = instance["terminals"]
    source, target = terminals[0], terminals[1]
    
    rep_nodes = instance.get("repairable_nodes", [])
    if not rep_nodes:
        rep_nodes = [n for n in nodes if n not in terminals]
        
    # Mapping entre ID du noeud et son index dans la liste x/y
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    # On récupère les bonnes probas pour les bons noeuds
    p_fail = np.array([instance["x"][node_to_idx[n]][0] for n in rep_nodes])
    pi_star = np.array([instance["y"][node_to_idx[n]] for n in rep_nodes])
    
    G = nx.DiGraph() if instance["graph"].get("is_directed", True) else nx.Graph()
    G.add_edges_from(edges)
    
    states = np.ones((n_sims, len(rep_nodes)), dtype=bool)
    
    # NOUVEAU : On utilise un compteur de temps de panne au lieu d'un booléen définitif
    downtime_counts = np.zeros(n_sims, dtype=float)
    
    for step in range(H):
        rands = np.random.rand(n_sims, len(rep_nodes))
        to_down = states & (rands < p_fail)
        to_up = (~states) & (rands < pi_star)
        
        # Mise à jour des états
        states = (states & ~to_down) | to_up
        
        # NOUVEAU : On évalue TOUTES les simulations à chaque pas de temps
        for idx in range(n_sims):
            up_rep_nodes = [rep_nodes[i] for i, is_up in enumerate(states[idx]) if is_up]
            
            # Si le réseau est déconnecté à l'instant t, on ajoute 1 au compteur de pannes
            if not fast_is_connected(G, source, target, up_rep_nodes):
                downtime_counts[idx] += 1
                
        # /!\ On a retiré le 'break' ici. Si le réseau tombe en panne, 
        # il peut tout à fait être réparé au pas de temps t+1 grâce au budget !

    # Le score final est la moyenne des temps d'indisponibilité, ramenée à l'horizon H (pourcentage)
    return np.mean(downtime_counts) / H

if __name__ == "__main__":
    print("Chargement du dataset...")
    # Remplace par le nom de ton fichier de test
    dataset_file = "dataset_hybrid_mesh_sp_er_v2_1000.json"
    
    with open(dataset_file, "r") as f: 
        data = json.load(f)
        
    all_instances = data["instances"]
    
    # On valide sur un échantillon (taille adaptée à la taille du dataset)
    if len(all_instances) > 5000:
        sample_size = min(5000, len(all_instances))
        validation_sample = random.sample(all_instances, sample_size)
    else:
        sample_size = len(all_instances)
        validation_sample = all_instances
    
    print(f"Démarrage de la simulation Monte-Carlo sur {sample_size}/{len(all_instances)} graphes...")
    print("Chaque graphe sera simulé 10,000 fois.")
    print("=" * 60)
    
    errors = []
    
    for idx, inst in enumerate(tqdm(validation_sample)):
        j_emp = simulate_monte_carlo(inst, n_sims=10000)
        j_ana = inst["J_star"]
        
        # Calcul de l'erreur absolue et relative
        error = abs(j_emp - j_ana)
        errors.append(error)
        
        # Affichage d'un debug tous les 500 graphes
        if idx % 500 == 0 and idx > 0:
            mae_so_far = np.mean(errors)
            tqdm.write(f"Graphe {idx} | J*: {j_ana:.4f} | MC: {j_emp:.4f} | Erreur: {error:.4f} | MAE: {mae_so_far:.4f}")

    errors_array = np.array(errors)
    mae = np.mean(errors_array)
    mse = np.mean(errors_array**2)
    rmse = np.sqrt(mse)
    
    print("\n" + "="*60)
    print(f"VALIDATION MONTE-CARLO TERMINÉE")
    print(f"Instances validées: {sample_size}/{len(all_instances)}")
    print("-" * 60)
    print(f"Erreur Absolue Moyenne (MAE):      {mae:.6f}")
    print(f"Erreur Quadratique Moyenne (MSE):  {mse:.6f}")
    print(f"Racine MSE (RMSE):                 {rmse:.6f}")
    print(f"Min/Max erreur:                    {errors_array.min():.6f} / {errors_array.max():.6f}")
    print(f"Écart-type:                        {np.std(errors_array):.6f}")
    print("="*60)