import networkx as nx
import matplotlib.pyplot as plt

# On importe ton générateur (assure-toi que le nom du fichier est bien generate_mesh1.py)
from generate_mesh1 import generate_mesh_instance

def plot_and_save_graph(G, record, filename="mesh_graph_visualization.png"):
    """
    Génère une visualisation claire du graphe pour le rapport et la sauvegarde en PNG.
    """
    # 1. Définition de la disposition spatiale (layout)
    # spring_layout donne un rendu aéré et naturel pour les maillages
    pos = nx.spring_layout(G, seed=42)
    
    # 2. Séparation des catégories de noeuds
    source_node = record["terminals"][0]
    target_node = record["terminals"][1]
    repairable_nodes = record["repairable_nodes"]
    
    plt.figure(figsize=(10, 8))
    
    # 3. Dessin des noeuds avec des couleurs distinctives
    nx.draw_networkx_nodes(G, pos, nodelist=[source_node], 
                           node_color='limegreen', node_size=800, label='Source (Entrée)')
    nx.draw_networkx_nodes(G, pos, nodelist=[target_node], 
                           node_color='crimson', node_size=800, label='Cible (Sortie)')
    nx.draw_networkx_nodes(G, pos, nodelist=repairable_nodes, 
                           node_color='skyblue', node_size=600, label='Routeurs (Réparables)')
    
    # 4. Dessin des arêtes (flèches directionnelles)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=15, 
                           edge_color='gray', connectionstyle='arc3,rad=0.1')
    
    # 5. Ajout des numéros de noeuds
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
    
    # 6. Finitions esthétiques
    plt.title(f"Topologie du réseau : {record['topology']}\n(Mise en évidence des Terminaux et des cycles)", fontsize=14)
    plt.legend(scatterpoints=1, loc='upper left', bbox_to_anchor=(1, 1))
    plt.axis('off') # On cache le cadre
    
    # 7. Sauvegarde en haute résolution (dpi=300 parfait pour un rapport)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() # On ferme la figure pour libérer la RAM

if __name__ == "__main__":
    # Génération avec source/target aléatoires (seed différent pour chaque exécution)
    import random
    seed = random.randint(0, 10000)
    G, record = generate_mesh_instance(m=3, n=4, seed=seed)
    
    # Création de l'image
    output_name = f"mesh_3x4_random_seed{seed}.png"
    plot_and_save_graph(G, record, filename=output_name)
    
    # Affichage des informations
    print(f"Seed utilisé : {seed}")
    print(f"Source : noeud {record['terminals'][0]}")
    print(f"Target : noeud {record['terminals'][1]}")
    print(f"Image générée avec succès : {output_name}")