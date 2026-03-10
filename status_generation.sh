#!/bin/bash
# Vérifier le statut de la génération

PROJECT_DIR="/users/local/l24virbe/Projet_ML"
OUTPUT_LOG="${PROJECT_DIR}/generation_output_v2_1000.log"
PROGRESS_LOG="${PROJECT_DIR}/generation_progress.log"
PID_FILE="${PROJECT_DIR}/generation_v2_1000.pid"
SESSION_NAME="generation_v2_1000"

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}📊 STATUT GÉNÉRATION V2_1000${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Vérifier screen
if command -v screen &> /dev/null; then
    SCREEN_SESSION=$(screen -ls 2>/dev/null | grep "$SESSION_NAME" || true)
    if [ ! -z "$SCREEN_SESSION" ]; then
        echo -e "${GREEN}✓ Session screen détectée${NC}"
        echo "  $SCREEN_SESSION"
        echo ""
        echo -e "${YELLOW}Reprendre :${NC}"
        echo "  screen -r $SESSION_NAME"
        echo ""
    fi
fi

# Vérifier tmux
if command -v tmux &> /dev/null; then
    TMUX_SESSION=$(tmux list-sessions 2>/dev/null | grep "$SESSION_NAME" || true)
    if [ ! -z "$TMUX_SESSION" ]; then
        echo -e "${GREEN}✓ Session tmux détectée${NC}"
        echo "  $TMUX_SESSION"
        echo ""
        echo -e "${YELLOW}Reprendre :${NC}"
        echo "  tmux attach-session -t $SESSION_NAME"
        echo ""
    fi
fi

# Vérifier processus nohup
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Processus actif${NC}"
        echo "  PID: $PID"
        echo ""
    fi
fi

# Afficher les logs
echo -e "${YELLOW}📋 Dernier(s) log(s) :${NC}"
echo ""

if [ -f "$PROGRESS_LOG" ]; then
    echo -e "${YELLOW}(progress_log)${NC}"
    tail -10 "$PROGRESS_LOG" | sed 's/^/   /'
    echo ""
fi

if [ -f "$OUTPUT_LOG" ]; then
    echo -e "${YELLOW}(output_log - dernières 5 lignes)${NC}"
    tail -5 "$OUTPUT_LOG" | sed 's/^/   /'
fi

# Vérifier dataset
DATASET="${PROJECT_DIR}/dataset_hybrid_mesh_sp_er_v2_1000.json"
if [ -f "$DATASET" ]; then
    SIZE=$(ls -lh "$DATASET" | awk '{print $5}')
    echo ""
    echo -e "${YELLOW}📁 Dataset :${NC}"
    echo "  Taille : $SIZE"
fi

echo ""
echo -e "${YELLOW}📝 Commandes utiles :${NC}"
echo "  tail -f generation_output_v2_1000.log  # Voir en direct"
echo "  tail -f generation_progress.log    # Voir le log de progression"
echo "  ./stop_generation.sh               # Arrêter proprement"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
