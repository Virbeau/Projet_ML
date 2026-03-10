#!/bin/bash
# Script de lancement résilient pour génération v2_1000 en arrière-plan
# Fonctionne même après fermeture de la VM/SSH

set -e

PROJECT_DIR="/users/local/l24virbe/Projet_ML"
PYTHON_ENV="${PROJECT_DIR}/env_projet/bin/activate"
OUTPUT_LOG="${PROJECT_DIR}/generation_output_v2_1000.log"
PROGRESS_LOG="${PROJECT_DIR}/generation_progress.log"
PID_FILE="${PROJECT_DIR}/generation_v2_1000.pid"
SESSION_NAME="generation_v2_1000"

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# En-tête
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 LANCEMENT RÉSILIENT - DATASET V2 1000 GRAPHES (détaché)${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Vérifications
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}❌ Dossier projet non trouvé${NC}"
    exit 1
fi

if [ ! -f "$PYTHON_ENV" ]; then
    echo -e "${RED}❌ Environnement Python non trouvé${NC}"
    exit 1
fi

# Vérifier si déjà en cours
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Génération déjà en cours (PID: $OLD_PID)${NC}"
        echo "Arrêtez avec: kill $OLD_PID"
        exit 1
    fi
fi

cd "$PROJECT_DIR"

# Vérifier method de détachement disponible
USE_SCREEN=false
USE_TMUX=false

if command -v screen &> /dev/null; then
    USE_SCREEN=true
    echo -e "${GREEN}✓ screening détecté (method 1)${NC}"
elif command -v tmux &> /dev/null; then
    USE_TMUX=true
    echo -e "${GREEN}✓ tmux détecté (method 2)${NC}"
else
    echo -e "${YELLOW}⚠️  screen/tmux non disponible → fallback nohup${NC}"
fi

echo ""
echo -e "${YELLOW}📊 Configuration :${NC}"
echo "  • Instances : 1 000 (1/3 MESH + 1/3 SP + 1/3 ER)"
echo "  • Sortie dataset : dataset_hybrid_mesh_sp_er_v2_1000.json"
echo "  • Logs :"
echo "    - Output : $OUTPUT_LOG"
echo "    - Progress : $PROGRESS_LOG"
echo "  • PID file : $PID_FILE"
echo ""

# MÉTHODE 1 : SCREEN (meilleure option)
if [ "$USE_SCREEN" = true ]; then
    echo -e "${GREEN}Lancement avec screen (détachable)...${NC}"
    
    screen -dmS $SESSION_NAME -c /dev/null bash -c "
        source $PYTHON_ENV
        cd $PROJECT_DIR
        echo \$\$ > $PID_FILE
        python main_production.py >> $OUTPUT_LOG 2>&1
        rm -f $PID_FILE
    "
    
    SCREEN_PID=$(pgrep -f "SCREEN.*$SESSION_NAME" | head -1)
    if [ ! -z "$SCREEN_PID" ]; then
        echo -e "${GREEN}✓ Session détachée créée${NC}"
        echo "  Session : $SESSION_NAME"
        echo "  PID : $SCREEN_PID"
        echo ""
        echo -e "${YELLOW}📋 Pour reprendre la session :${NC}"
        echo "  screen -r $SESSION_NAME"
        echo ""
        echo -e "${YELLOW}📋 Pour voir en direct :${NC}"
        echo "  tail -f $OUTPUT_LOG"
        echo ""
    else
        echo -e "${RED}❌ Erreur lors du lancement avec screen${NC}"
        exit 1
    fi

# MÉTHODE 2 : TMUX
elif [ "$USE_TMUX" = true ]; then
    echo -e "${GREEN}Lancement avec tmux (détachable)...${NC}"
    
    tmux new-session -d -s $SESSION_NAME -c "$PROJECT_DIR" bash -c "
        source $PYTHON_ENV
        echo \$\$ > $PID_FILE
        python main_production.py | tee $OUTPUT_LOG
        rm -f $PID_FILE
    "
    
    TMUX_PID=$(pgrep -f "tmux.*$SESSION_NAME" | head -1)
    if [ ! -z "$TMUX_PID" ]; then
        echo -e "${GREEN}✓ Session détachée créée${NC}"
        echo "  Session : $SESSION_NAME"
        echo "  PID : $TMUX_PID"
        echo ""
        echo -e "${YELLOW}📋 Pour reprendre la session :${NC}"
        echo "  tmux attach-session -t $SESSION_NAME"
        echo ""
        echo -e "${YELLOW}📋 Pour voir en direct :${NC}"
        echo "  tail -f $OUTPUT_LOG"
        echo ""
    else
        echo -e "${RED}❌ Erreur lors du lancement avec tmux${NC}"
        exit 1
    fi

# MÉTHODE 3 : NOHUP (fallback)
else
    echo -e "${GREEN}Lancement avec nohup (fallback)...${NC}"
    
    nohup bash -c "
        source $PYTHON_ENV
        cd $PROJECT_DIR
        echo \$\$ > $PID_FILE
        python main_production.py >> $OUTPUT_LOG 2>&1
        rm -f $PID_FILE
    " > /dev/null 2>&1 &
    
    BG_PID=$!
    echo "$BG_PID" > "$PID_FILE"
    
    echo -e "${GREEN}✓ Processus lancé en arrière-plan${NC}"
    echo "  PID : $BG_PID"
    echo ""
fi

echo -e "${YELLOW}📝 Notes importantes :${NC}"
echo "  1. vous pouvez fermer votre terminal → génération continue"
echo "  2. Les logs sont sauvegardés en direct"
echo "  3. La génération s'ajuste au redémarrage même si interrompue"
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Lancement réussi - Génération en cours${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
