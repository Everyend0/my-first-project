#!/usr/bin/env bash
# Full pipeline: drawing -> segmentation -> 3D meshes -> OBJ/MTL/.blend -> renders -> sheets
# usage: pipeline/run_all.sh [banana_legs banana_arms]  (env: SAMPLES, RES, SKIP_BUILD=1)
set -euo pipefail
cd "$(dirname "$0")/.."
SAMPLES=${SAMPLES:-64}
RES=${RES:-1400}
NAMES=${*:-banana_legs banana_arms}
mkdir -p work output
for n in $NAMES; do
  echo "=== $n  $(date +%T)"
  if [ "${SKIP_BUILD:-0}" != "1" ]; then
    python3 pipeline/segment.py "src/$n.png" "work/$n.seg.npz" "work/$n.seg.png" > "work/$n.segment.log"
    python3 pipeline/build_mesh.py "$n" "work/$n.seg.npz" "work/$n.mesh.npz" > "work/$n.build.log"
  fi
  rm -f output/$n/renders/*.png
  python3 pipeline/scene.py "$n" --samples "$SAMPLES" --res "$RES" --ortho-check 2>&1 \
    | grep -E "exported|rendered|Error|Traceback" || true
  python3 pipeline/compare.py "src/$n.png" "output/$n/renders/${n}_front_ortho_match.png" \
    "output/$n/${n}_overlay_check.png" | tee "output/$n/${n}_fidelity.json"
  python3 pipeline/sheet.py "$n"
done
# Unity colliders are fitted to the exported OBJs of both bananas
if [ -f output/banana_legs/banana_legs.obj ] && [ -f output/banana_arms/banana_arms.obj ]; then
  python3 pipeline/colliders.py
fi
echo "=== done $(date +%T)"
