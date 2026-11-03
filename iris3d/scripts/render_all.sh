#!/usr/bin/env bash
# Render every 4K frame and encode the three 4K 30 fps videos.
#
#   front_tilt : front view from 15 deg above   (1 s loop rendered, played 3x)
#   front      : straight-on orthographic view  (1 s loop rendered, played 3x)
#   orbit      : camera sways +-20 deg over 3 s (all 90 frames rendered)
#
# The animation is an exact 1-second loop, so for the two fixed cameras the
# 30 rendered frames ARE frames 31-60 and 61-90 as well (verify_iris.py
# checks the mesh repeats to ~1e-7 m). Re-running skips finished frames.
set -euo pipefail
cd "$(dirname "$0")/.."
BL=scripts/blender.sh
FR=build/frames
mkdir -p "$FR" renders

render() {  # camera start end outdir
  "$BL" -b model/iris.blend --python scripts/render_iris.py -- \
    --camera "$1" --start "$2" --end "$3" --out "$4" 2>&1 | grep -E "^(FRAME|DONE)|Error" || true
}

case "${1:-all}" in
  front_tilt|all) render CAM_front_tilt 1 30 "$FR/front_tilt" ;;&
  front|all)      render CAM_front      1 30 "$FR/front" ;;&
  orbit|all)      render CAM_orbit      1 90 "$FR/orbit" ;;&
  encode|all)     scripts/encode.sh ;;
  *) ;;
esac
