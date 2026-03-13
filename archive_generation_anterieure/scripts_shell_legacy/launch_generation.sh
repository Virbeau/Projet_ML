#!/bin/bash
# Script de lancement pour génération 50k avec monitorings

set -e  # Exit on error

# Configuration
PROJECT_DIR="/users/local/l24virbe/Projet_ML"
PYTHON_ENV="${PROJECT_DIR}/env_projet/bin/activate"
OUTPUT_LOG="generation_output.log"
PROGRESS_LOG="generation_progress.log"

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# En-tête
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 LANCEMENT GÉNÉRATION 50 000 GRAPHES${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Vérifier que le projet existe
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}❌ Dossier projet non trouvé: $PROJECT_DIR${NC}"
    exit 1
fi

cd "$PROJECT_DIR"

# Vérifier Python
if [ ! -f "$PYTHON_ENV" ]; then
    echo -e "${RED}❌ Environment Python non trouvé${NC}"
    exit 1
fi

# Activation environnement
source "$PYTHON_ENV"
echo -e "${GREEN}✓ Environnement activé${NC}"
echo ""

# Afficher configuration
echo -e "${YELLOW}📊 Configuration :${NC}"
echo "  • Projet: $PROJECT_DIR"
echo "  • Instances: 50 000 (1/3 MESH + 1/3 SP + 1/3 ER)"
echo "  • Temps estimé: ~33h"
echo "  • Logs:"
echo "    - Output: $OUTPUT_LOG"
echo "    - Progress: $PROGRESS_LOG"
echo ""

# Démarrer la génération
echo -e "${BLUE}Démarrage...${NC}"
echo "⏱️  Notez l'heure : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Lancer avec output duplicate (terminal + fichier)
python main_production.py 2>&1 | tee "$OUTPUT_LOG"

# Statistiques finales
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ GÉNÉRATION COMPLÈTE${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}📁 Fichiers générés :${NC}"
ls -lh dataset_hybrid_mesh_sp_er.json "$PROGRESS_LOG" 2>/dev/null || true
echo ""
echo -e "${YELLOW}📋 Résultats finaux (dernières lignes du log):${NC}"
tail -10 "$PROGRESS_LOG" 2>/dev/null || true
echo ""
echo -e "${GREEN}✓ Fin : $(date '+%Y-%m-%d %H:%M:%S')${NC}"
