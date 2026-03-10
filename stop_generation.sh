#!/bin/bash
# Arrêter proprement la génération

PROJECT_DIR="/users/local/l24virbe/Projet_ML"
PID_FILE="${PROJECT_DIR}/generation_v2_1000.pid"
SESSION_NAME="generation_v2_1000"

# Couleurs
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${RED}🛑 ARRÊT DE LA GÉNÉRATION${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

STOPPED=false

# Arrêter screen
if command -v screen &> /dev/null; then
    SCREEN_PID=$(screen -ls 2>/dev/null | grep "$SESSION_NAME" | grep -oE '\b[0-9]+\.' | cut -d. -f1)
    if [ ! -z "$SCREEN_PID" ]; then
        echo -e "${YELLOW}Arrêt de la session screen...${NC}"
        screen -X -S $SESSION_NAME quit 2>/dev/null || true
        sleep 2
        echo -e "${GREEN}✓ Session screen fermée${NC}"
        STOPPED=true
    fi
fi

# Arrêter tmux
if command -v tmux &> /dev/null; then
    if tmux list-sessions 2>/dev/null | grep -q "$SESSION_NAME"; then
        echo -e "${YELLOW}Arrêt de la session tmux...${NC}"
        tmux kill-session -t $SESSION_NAME 2>/dev/null || true
        sleep 2
        echo -e "${GREEN}✓ Session tmux fermée${NC}"
        STOPPED=true
    fi
fi

# Arrêter processus nohup
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}Arrêt du processus (PID: $PID)...${NC}"
        
        # Graceful shutdown : SIGTERM
        kill -TERM "$PID" 2>/dev/null || true
        
        # Attendre 5s
        COUNTER=0
        while ps -p "$PID" > /dev/null 2>&1 && [ $COUNTER -lt 50 ]; do
            sleep 0.1
            COUNTER=$((COUNTER + 1))
        done
        
        # Force kill si toujours actif
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}  Force kill (SIGKILL)...${NC}"
            kill -KILL "$PID" 2>/dev/null || true
        fi
        
        echo -e "${GREEN}✓ Processus arrêté${NC}"
        rm -f "$PID_FILE"
        STOPPED=true
    fi
fi

# Arrêter tous les python main_production.py
PYTHON_PIDS=$(pgrep -f "python.*main_production.py" || true)
if [ ! -z "$PYTHON_PIDS" ]; then
    echo -e "${YELLOW}Arrêt des processus Python restants...${NC}"
    echo "$PYTHON_PIDS" | while read PID; do
        kill -TERM "$PID" 2>/dev/null || true
    done
    sleep 2
    # Force kill si nécessaire
    REMAINING=$(pgrep -f "python.*main_production.py" || true)
    if [ ! -z "$REMAINING" ]; then
        echo "$REMAINING" | while read PID; do
            kill -KILL "$PID" 2>/dev/null || true
        done
    fi
    echo -e "${GREEN}✓ Processus Python arrêtés${NC}"
    STOPPED=true
fi

if [ "$STOPPED" = false ]; then
    echo -e "${YELLOW}⚠️  Aucun processus de génération trouvé${NC}"
fi

echo ""
echo -e "${YELLOW}📁 Fichiers sauvegardés :${NC}"
ls -lh "${PROJECT_DIR}/dataset_hybrid_mesh_sp_er_v2_1000.json" 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  [Pas encore de dataset]"
ls -lh "${PROJECT_DIR}/generation_progress.log" 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || true

echo ""
echo -e "${GREEN}✅ Arrêt gracieux complété${NC}"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
