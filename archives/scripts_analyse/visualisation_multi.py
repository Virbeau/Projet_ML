import networkx as nx
import matplotlib.pyplot as plt
import random
import os
import numpy as np

# Imports des trois générateurs
from generate_mesh1 import generate_mesh_instance
from generate_sp1 import generate_sp_instance
from generate_er import generate_er_instance

def get_mesh_layout(G, m, n):
    """
    Crée un layout basé sur la structure de grille pour les graphes MESH.
    Les positions suivent exactement la structure m×n.
    """
    pos = {}
    nodes = sorted(G.nodes())
    for i, node in enumerate(nodes):
        row = i // n
        col = i % n
        pos[node] = (col * 2, -row * 2)  # Espacement pour clarté
    return pos

def get_sp_layout(G):
    """
    Utilise l'algorithme Kamada-Kawai pour une meilleure distribution.
    Plus stable et lisible que spring_layout pour les graphes complexes.
    """
    return nx.kamada_kawai_layout(G)

def plot_graph(G, record, ax, title, graph_type, m=None, n=None):
    """
    Dessine un graphe sur un subplot matplotlib avec un layout optimisé.
    
    Args:
        graph_type: "mesh" ou "sp"
        m, n: dimensions pour les graphes mesh
    """
    # Sélectionner le meilleur layout selon le type de graphe
    if graph_type == "mesh":
        pos = get_mesh_layout(G, m, n)
    else:  # sp
        pos = get_sp_layout(G)
    
    # Séparation des catégories de nœuds
    source_node = record["terminals"][0]
    target_node = record["terminals"][1]
    repairable_nodes = record["repairable_nodes"]
    
    # === DESSIN DES ARÊTES EN PREMIER (fond) ===
    nx.draw_networkx_edges(G, pos, 
                           arrowstyle='-|>', 
                           arrowsize=20,
                           edge_color='#555555', 
                           alpha=0.5, 
                           ax=ax,
                           width=2.5,
                           connectionstyle='arc3,rad=0.1')
    
    # === DESSIN DES NŒUDS (avec meilleur contraste) ===
    # Source en vert vif
    nx.draw_networkx_nodes(G, pos, nodelist=[source_node], 
                           node_color='#00DD00', 
                           node_size=1200, 
                           ax=ax, 
                           edgecolors='#005500',
                           linewidths=3)
    
    # Target en rouge vif
    nx.draw_networkx_nodes(G, pos, nodelist=[target_node], 
                           node_color='#FF0000', 
                           node_size=1200, 
                           ax=ax,
                           edgecolors='#660000',
                           linewidths=3)
    
    # Réparables en bleu
    nx.draw_networkx_nodes(G, pos, nodelist=repairable_nodes, 
                           node_color='#0099FF', 
                           node_size=900, 
                           ax=ax,
                           edgecolors='#003366',
                           linewidths=2.5)
    
    # === LABELS DES NŒUDS (plus gros et lisibles) ===
    labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels,
                           font_size=10, 
                           font_weight="bold", 
                           font_color="white",
                           ax=ax)
    
    # === TITRE ET INFOS ===
    n_nodes = len(G.nodes())
    n_edges = len(G.edges())
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15, color='#222222')
    
    # Info en bas du graph
    info_text = f"{n_nodes} nœuds  •  {n_edges} arêtes  •  Source: {source_node}  •  Target: {target_node}"
    ax.text(0.5, -0.12, info_text, transform=ax.transAxes,
            fontsize=9, ha='center', style='italic', color='#555555')
    
    ax.axis('off')
    # Légende minimaliste
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#00DD00', edgecolor='#005500', label='Source'),
        Patch(facecolor='#FF0000', edgecolor='#660000', label='Target'),
        Patch(facecolor='#0099FF', edgecolor='#003366', label='Réparables')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, 
              framealpha=0.95, edgecolor='#999999')


def generate_and_visualize_topologies():
    """
    Génère et visualise 9 graphes aléatoires :
    - 3 graphes MESH de tailles variées
    - 3 graphes SÉRIE-PARALLÈLE
    - 3 graphes ERDŐS-RÉNYI avec différentes densités
    """
    print("=" * 70)
    print("GÉNÉRATION DE 9 GRAPHES - COMPARAISON DES 3 TOPOLOGIES")
    print("=" * 70)
    
    # Configuration des graphes à générer
    mesh_configs = [
        (2, 3, "Mesh 2×3"),
        (3, 3, "Mesh 3×3"),
        (2, 4, "Mesh 2×4"),
    ]
    
    sp_configs = [
        (3, "SP-3 réparables"),
        (5, "SP-5 réparables"),
        (7, "SP-7 réparables"),
    ]
    
    er_configs = [
        (6, 0.3, "ER (n=6, p=0.3)"),
        (8, 0.25, "ER (n=8, p=0.25)"),
        (10, 0.25, "ER (n=10, p=0.25)"),
    ]
    
    # Création de la figure avec 3 lignes et 3 colonnes
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Visualisation Comparative : 3 Topologies Hybrides (9 instances)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Génération des 3 graphes MESH (première ligne)
    print("\n📐 Génération des graphes MESH...")
    for idx, (m, n, label) in enumerate(mesh_configs):
        seed = random.randint(0, 10000)
        G, record = generate_mesh_instance(m=m, n=n, seed=seed)
        
        n_nodes = len(G.nodes())
        n_edges = len(G.edges())
        title = f"{label}\n({m}×{n} = {n_nodes} nœuds)"
        
        plot_graph(G, record, axes[0, idx], title, graph_type="mesh", m=m, n=n)
        print(f"   [{idx+1}] {label} - Source: {record['terminals'][0]}, Target: {record['terminals'][1]}")
    
    # Génération des 3 graphes SP (deuxième ligne)
    print("\n🔄 Génération des graphes SÉRIE-PARALLÈLE...")
    for idx, (num_rep, label) in enumerate(sp_configs):
        seed = random.randint(0, 10000)
        G, record = generate_sp_instance(num_repairable=num_rep, seed=seed)
        
        n_nodes = len(G.nodes())
        n_edges = len(G.edges())
        title = f"{label}\n({n_nodes} nœuds)"
        
        plot_graph(G, record, axes[1, idx], title, graph_type="sp")
        print(f"   [{idx+4}] {label}")
    
    # Génération des 3 graphes ER (troisième ligne)
    print("\n🌐 Génération des graphes ERDŐS-RÉNYI...")
    for idx, (num_nodes, p, label) in enumerate(er_configs):
        seed = random.randint(0, 10000)
        G, record = generate_er_instance(num_nodes=num_nodes, p=p, seed=seed)
        
        n_nodes = len(G.nodes())
        n_edges = len(G.edges())
        title = f"{label}\n({n_nodes} nœuds, {n_edges} arêtes)"
        
        plot_graph(G, record, axes[2, idx], title, graph_type="sp")
        print(f"   [{idx+7}] {label}")
    
    # Ajustement de l'espacement
    plt.tight_layout(rect=[0, 0, 1, 0.993], pad=1.5, w_pad=1.2, h_pad=2.2)
    
    # Sauvegarde de l'image
    output_file = "graphes_visualisation_3_topologies.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✅ Visualisation générée : {output_file}")
    print(f"📐 Ligne 1 : MESH structurés (grille alignée)")
    print(f"🔄 Ligne 2 : Série-Parallèle (Kamada-Kawai layout)")
    print(f"🌐 Ligne 3 : Erdős-Rényi (aléatoires, Kamada-Kawai)")
    print(f"📊 Format : 18×15 pouces, 300 DPI, haute résolution")
    print("=" * 70)


if __name__ == "__main__":
    generate_and_visualize_topologies()
