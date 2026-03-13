#!/bin/bash
# Affiche le setup complet et les fichiers prêts

PROJECT_DIR="/users/local/l24virbe/Projet_ML"
cd "$PROJECT_DIR"

# Couleurs
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}📊 SETUP COMPLET - GÉNÉRATION 50K GRAPHES${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Scripts
echo -e "${CYAN}🚀 SCRIPTS (exécutables)${NC}"
echo ""
for script in launch_detached.sh status_generation.sh stop_generation.sh monitor_generation.sh launch_generation.sh; do
    if [ -f "$script" ]; then
        SIZE=$(ls -lh "$script" | awk '{print $5}')
        echo -e "  ${GREEN}✓${NC} $script ($SIZE)"
    fi
done
echo ""

# Code sources
echo -e "${CYAN}📝 CODE SOURCES${NC}"
echo ""
for file in main_production.py generate_mesh1.py generate_sp1.py generate_er.py solver.py visualisation_multi.py validate_gnn_format.py; do
    if [ -f "$file" ]; then
        LINES=$(wc -l < "$file")
        echo -e "  ${GREEN}✓${NC} $file ($LINES lignes)"
    fi
done
echo ""

# Documentation
echo -e "${CYAN}📚 DOCUMENTATION${NC}"
echo ""
for doc in START_HERE.md CHECKLIST_FINAL.md SETUP_COMPLET.md GUIDE_MACHINE_VIRTUELLE.md README_GENERATION.md GUIDE_GENERATION_50K.md; do
    if [ -f "$doc" ]; then
        SIZE=$(ls -lh "$doc" | awk '{print $5}')
        echo -e "  ${GREEN}✓${NC} $doc ($SIZE)"
    fi
done
echo ""

# Statistiques
echo -e "${CYAN}📊 CONFIGURATION${NC}"
echo ""
echo -e "  ${GREEN}✓${NC} n_instances : $(grep -oP 'n_instances = \K\d+' main_production.py | head -1)"
echo -e "  ${GREEN}✓${NC} Topologies : 1/3 MESH + 1/3 SP + 1/3 ER"
echo -e "  ${GREEN}✓${NC} Tailles : 4-12 nœuds par graphe"
echo -e "  ${GREEN}✓${NC} Temps estimé : ~33 heures"
echo -e "  ${GREEN}✓${NC} Output : dataset_hybrid_mesh_sp_er.json (~250 MB)"
echo ""

# Vérifications
echo -e "${CYAN}✅ VÉRIFICATIONS${NC}"
echo ""

# Python env
if [ -f "env_projet/bin/python" ]; then
    PY_VERSION=$(env_projet/bin/python --version)
    echo -e "  ${GREEN}✓${NC} Python : $PY_VERSION"
fi

# Dépendances
echo -n "  ${GREEN}✓${NC} NetworkX : "
env_projet/bin/python -c "import networkx; print(networkx.__version__)" 2>/dev/null || echo "non trouvé"

echo -n "  ${GREEN}✓${NC} NumPy : "
env_projet/bin/python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "non trouvé"

echo -n "  ${GREEN}✓${NC} tqdm : "
env_projet/bin/python -c "import tqdm; print('installé')" 2>/dev/null || echo "non trouvé"

echo ""

# Quick start
echo -e "${CYAN}🚀 QUICK START${NC}"
echo ""
echo -e "  ${YELLOW}Lancer :${NC}"
echo "    ./launch_detached.sh"
echo ""
echo -e "  ${YELLOW}Vérifier l'état :${NC}"
echo "    ./status_generation.sh"
echo ""
echo -e "  ${YELLOW}Voir en direct :${NC}"
echo "    tail -f generation_output.log"
echo ""
echo -e "  ${YELLOW}Arrêter :${NC}"
echo "    ./stop_generation.sh"
echo ""

# Capacités
echo -e "${CYAN}💡 CAPACITÉS${NC}"
echo ""
echo -e "  ${GREEN}✓${NC} Lancer sur VM et fermer (persistance)"
echo -e "  ${GREEN}✓${NC} Monitorer en temps réel"
echo -e "  ${GREEN}✓${NC} Arrêter proprement"
echo -e "  ${GREEN}✓${NC} Reprendre la session"
echo -e "  ${GREEN}✓${NC} Dataset GNN-ready"
echo -e "  ${GREEN}✓${NC} PyTorch Geometric compatible"
echo ""

# Status final
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ STATUS : PRÊT À LANCER${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""
