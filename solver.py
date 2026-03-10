# --- Imports ---
import numpy as np
import networkx as nx
from numba import jit

# On enlève matplotlib car on fait tourner ça sur un serveur/VM en tâche de fond
np.set_printoptions(precision=3, suppress=True)


# --- Fonctions Utilitaires Manquantes ---
@jit(nopython=True)
def int_to_bits(s, m):
    """Version ultra-rapide avec Numba JIT (manipulation de bits pure)"""
    bits = np.zeros(m, dtype=np.int32)
    for i in range(m):
        bits[m - 1 - i] = (s >> i) & 1
    return bits


# --- Critère d'évaluation (Adapté pour les Graphes Orientés / Mesh) ---
def phi_terminal_connectivity(G_res, terminals):
    """
    Vérifie si la Source (terminals[0]) peut toujours joindre la Cible (terminals[-1]).
    G_res est le sous-graphe contenant uniquement les noeuds qui ne sont PAS en panne.
    """
    source = terminals[0]
    target = terminals[-1]
    
    # Si la source ou la cible est crashée (ne devrait pas arriver, ils sont parfaits)
    if source not in G_res or target not in G_res:
        return 0
        
    try:
        # nx.has_path gère parfaitement les graphes orientés (DAGs et Mesh)
        return 1 if nx.has_path(G_res, source, target) else 0
    except nx.NetworkXNoPath:
        return 0


def get_fail_mask(G, terminals, criterion, params):
    """
    Explore les 2^m états possibles. Renvoie un masque booléen où True = "Système en panne".
    """
    m = len(params["repairable_nodes"])
    S = 1 << m
    fail_mask = np.zeros(S, dtype=bool)

    for s in range(S):
        x = int_to_bits(s, m)
        
        # On garde les terminaux (toujours actifs)
        active_nodes = set(terminals)
        # On ajoute les noeuds réparables qui sont à l'état 1 (UP)
        for i, node in enumerate(params["repairable_nodes"]):
            if x[i] == 1: 
                active_nodes.add(node)
        
        # On crée le graphe résiduel
        G_res = G.subgraph(active_nodes)
        
        # On évalue le critère
        if criterion == "terminal_connectivity":
            is_up = phi_terminal_connectivity(G_res, terminals)
        else:
            is_up = 1 # Fallback au cas où
            
        # fail_mask = True si le réseau est mort (is_up == 0)
        fail_mask[s] = (is_up == 0)
        
    return fail_mask


# --- Moteur Markovien (Le coeur du prof, légèrement optimisé) ---
def build_transition_matrix(p_fail, r_repair):
    """
    Version 100% vectorisée NumPy. 
    Calcule le million de probabilités en 0.01 seconde au lieu de 15 minutes !
    """
    m = len(p_fail)
    S = 1 << m
    
    # On pré-calcule TOUS les états binaires d'un seul coup. Matrice de taille (S, m)
    all_bits = np.array([[(s >> (m - 1 - i)) & 1 for i in range(m)] for s in range(S)])
    
    # p1 : Probabilité que le noeud soit UP au tour suivant (taille S x m)
    p1 = np.where(all_bits == 1, 1.0 - p_fail, r_repair)
    p0 = 1.0 - p1
    
    P = np.zeros((S, S), dtype=float)
    
    # On calcule les lignes de la matrice par blocs entiers !
    for s in range(S):
        # Pour l'état de départ 's', on broadcast contre TOUS les états d'arrivée possibles
        probs_y = np.where(all_bits == 1, p1[s], p0[s])
        P[s] = np.prod(probs_y, axis=1) # Multiplication rapide sur la dimension des noeuds
        
    # Stabilisation numérique (normalisation des lignes)
    rs = P.sum(axis=1, keepdims=True)
    # Pour éviter la division par zéro
    P = np.divide(P, rs, out=np.zeros_like(P), where=rs!=0)
    
    # Si une ligne était vide (ne devrait pas arriver), on met 1 sur la diagonale
    zero_rows = (rs.flatten() == 0)
    P[zero_rows, zero_rows] = 1.0
    
    return P


def hitting_probability(P, fail_mask, H):
    S = P.shape[0]
    h = fail_mask.astype(float).copy() 
    traj = [h.copy()]
    for _ in range(H):
        h_next = fail_mask.astype(float) + (~fail_mask).astype(float) * (P @ h)
        h_next = np.clip(h_next, 0.0, 1.0)
        h = h_next
        traj.append(h.copy())
    return h, traj


def compute_objective_J(G, terminals, criterion, params, pi, H, fail_mask=None):
    """
    Calcule le score J (La probabilité que le réseau tombe en panne avant la fin du temps H).
    On veut MINIMISER cette probabilité.
    """
    if fail_mask is None:
        fail_mask = get_fail_mask(G, terminals, criterion, params)
        
    P = build_transition_matrix(params["p_fail"], pi)
    h, _ = hitting_probability(P, fail_mask, H)
    
    m = len(pi)
    start_state = (1 << m) - 1 # Au temps t=0, tous les noeuds marchent (111...1)
    
    return h[start_state]


# --- L'Optimiseur (Calcul du label pi*) ---
def projected_gradient_descent(G, terminals, criterion, params, H, B, lr=0.8, iters=25, eps=1e-3, verbose=False):
    """
    Trouve la meilleure allocation de budget (pi*) pour minimiser la panne.
    """
    m = len(params["repairable_nodes"])
    pi = np.zeros(m) # On part avec 0 budget
    c_cost = params["c_cost"]
    hist = []
    
    # On pré-calcule le fail_mask une seule fois pour gagner un temps énorme !
    fail_mask = get_fail_mask(G, terminals, criterion, params)
    
    for it in range(iters):
        # 1. Calcul du gradient (Différences finies)
        grad = np.zeros(m)
        J_current = compute_objective_J(G, terminals, criterion, params, pi, H, fail_mask)
        
        for i in range(m):
            pi_eps = pi.copy()
            pi_eps[i] = min(1.0, pi[i] + eps)
            J_eps = compute_objective_J(G, terminals, criterion, params, pi_eps, H, fail_mask)
            grad[i] = (J_eps - J_current) / eps
            
        # 2. Descente de gradient (J_current est une probabilité de panne, on veut la réduire)
        pi_new = pi - lr * grad
        
        # 3. Projection sur le budget (Somme des pi*cost <= B)
        pi_new = np.clip(pi_new, 0.0, 1.0)
        current_cost = np.sum(pi_new * c_cost)
        if current_cost > B:
            pi_new = pi_new * (B / current_cost) # On réduit proportionnellement
            
        pi = np.clip(pi_new, 0.0, 1.0)
        hist.append(J_current)
        
    return pi, hist


# --- Fonction Principale appelée par main.py ---
def solve_instance(G, terminals, criterion, params, H, B, seed=0, iters=25):
    # On lance l'optimisation
    pi_opt, hist = projected_gradient_descent(G, terminals, criterion, params, H=H, B=B, iters=iters)
    J = compute_objective_J(G, terminals, criterion, params, pi_opt, H=H)
    
    pi_by_node = {params["repairable_nodes"][i]: float(pi_opt[i]) for i in range(len(pi_opt))}
    for t in terminals:
        pi_by_node[t] = 0.0
        
  
    return pi_opt, J, pi_by_node, hist

