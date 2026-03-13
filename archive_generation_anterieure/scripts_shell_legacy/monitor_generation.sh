#!/bin/bash
# Script pour monitorer la génération en cours

PROJECT_DIR="/users/local/l24virbe/Projet_ML"
PROGRESS_LOG="${PROJECT_DIR}/generation_progress.log"
OUTPUT_LOG="${PROJECT_DIR}/generation_output_v2_1000.log"
DATASET_FILE="${PROJECT_DIR}/dataset_hybrid_mesh_sp_er_v2_1000.json"

# Couleurs
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Vérifier si une génération est en cours
if ! pgrep -f "python main_production.py" > /dev/null; then
    if [ -f "$PROGRESS_LOG" ]; then
        echo -e "${YELLOW}⚠️  Pas de génération en cours${NC}"
        echo -e "${GREEN}Dernier résultat :${NC}"
        tail -20 "$PROGRESS_LOG"
    else
        echo -e "${RED}❌ Aucune génération lancée${NC}"
    fi
    exit 0
fi

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}📊 MONITORING GÉNÉRATION V2_1000${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Affichage continu avec refresh
while true; do
    clear
    
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}📊 MONITORING GÉNÉRATION V2_1000${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Vérifier si processus toujours actif
    if ! pgrep -f "python main_production.py" > /dev/null; then
        echo -e "${GREEN}✅ Génération terminée${NC}"
        echo ""
        if [ -f "$PROGRESS_LOG" ]; then
            echo -e "${YELLOW}Résultats finaux :${NC}"
            tail -15 "$PROGRESS_LOG"
        fi
        exit 0
    fi
    
    # Affichage en temps réel
    echo -e "${YELLOW}⏱️  Processus actif (PID: $(pgrep -f 'python main_production.py'))${NC}"
    echo ""
    
    # Afficher les dernières lignes du output
    echo -e "${YELLOW}📝 Dernier output :${NC}"
    if [ -f "$OUTPUT_LOG" ]; then
        tail -5 "$OUTPUT_LOG" | sed 's/^/   /'
    fi
    echo ""
    
    # Stats du fichier log
    echo -e "${YELLOW}📊 Stats du log :${NC}"
    if [ -f "$PROGRESS_LOG" ]; then
        wc -l "$PROGRESS_LOG" | awk '{print "   Lignes : " $1}'
        ls -lh "$PROGRESS_LOG" | awk '{print "   Taille : " $5}'
    fi
    echo ""
    
    # Stats du dataset partiel
    if [ -f "$DATASET_FILE" ]; then
        FILE_SIZE=$(ls -lh "$DATASET_FILE" | awk '{print $5}')
        LINE_COUNT=$(grep -c '"topology_type"' "$DATASET_FILE" 2>/dev/null || echo "?")
        if [ "$LINE_COUNT" != "?" ]; then
            INSTANCES=$((LINE_COUNT))
            PROGRESS=$((INSTANCES * 100 / 1000))
            echo -e "${YELLOW}📁 Dataset partiel :${NC}"
            echo "   Instances : $INSTANCES/1000 ($PROGRESS%)"
            echo "   Taille : $FILE_SIZE"
        fi
    fi
    echo ""
    
    # Infos CPU/Mémoire
    echo -e "${YELLOW}⚙️  Ressources système :${NC}"
    ps aux | grep "python main_production.py" | grep -v grep | awk '{printf "   CPU: %5.1f%% | Mémoire: %5.1f MB\n", $3, $6/1024}' || echo "   [Processus en transition]"
    echo ""
    
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Refresh toutes les 30s (Ctrl+C pour quitter)${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    
    sleep 30
done
