#!/bin/bash
# Submit one independent PBS job per variant (4 jobs total, each on its own A100).
# Usage: bash scripts/submit_wxf_ablation.sh [STEPS] [BATCH_SIZE]
#
# Examples:
#   bash scripts/submit_wxf_ablation.sh              # 5000 steps, batch 4
#   bash scripts/submit_wxf_ablation.sh 10000 4

STEPS=${1:-5000}
BATCH=${2:-4}
SCRIPT=$(dirname "$0")/wxf_ablation_casper.sh

echo "Submitting 8 independent PBS jobs (one per variant)"
echo "  STEPS=${STEPS}  BATCH=${BATCH}"
echo ""

for V in v1 v3 "v3+grid" "v3+shift+grid" v2_full v2_no_moe v2_no_register v2_no_decoder_attn; do
    JID=$(qsub -N "wxf_${V//+/_}" \
               -v VARIANT="${V}",STEPS="${STEPS}",BATCH_SIZE="${BATCH}" \
               "${SCRIPT}")
    printf "  %-20s -> %s\n" "${V}" "${JID}"
done
