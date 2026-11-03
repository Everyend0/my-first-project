#!/usr/bin/env bash
# Render the 4K frames and encode the three 4K 30 fps videos.
#
#   scripts/render_all.sh                 # everything (~5 h on 4 CPU cores)
#   scripts/render_all.sh orbit 1 45      # one camera, a frame range
#   scripts/render_all.sh encode          # composite + encode only
#
# Cameras:
#   front_tilt : front view from 15 deg above   (1 s loop rendered, played 3x)
#   front      : straight-on orthographic view  (1 s loop rendered, played 3x)
#   orbit      : camera sways +-20 deg over 3 s (all 90 frames rendered)
#
# The animation is an exact 1-second loop (verify_iris.py: loop seam error 0),
# so for the two fixed cameras frames 31-90 are identical to frames 1-30.
# Each frame is two passes: the mascot at full 4K ('char') and the soft
# backdrop shadows at 25 % ('shadow'), composited over white by
# composite_iris.py. Re-running skips frames that are already done.
set -euo pipefail
cd "$(dirname "$0")/.."
BL=scripts/blender.sh
FR=build/frames

render() {  # name camera start end
  local name=$1 cam=$2 s=$3 e=$4
  "$BL" -b model/iris.blend --python scripts/render_iris.py -- --camera "$cam" \
      --start "$s" --end "$e" --pass shadow --samples 32 --out "$FR/$name/shadow" 2>&1 \
      | grep -E "^(FRAME|DONE)|Error" || true
  "$BL" -b model/iris.blend --python scripts/render_iris.py -- --camera "$cam" \
      --start "$s" --end "$e" --pass char --out "$FR/$name/char" 2>&1 \
      | grep -E "^(FRAME|DONE)|Error" || true
  python3 scripts/composite_iris.py "$FR/$name/char" "$FR/$name/shadow" "$FR/$name/final"
}

cam_of() {
  case $1 in
    front_tilt) echo CAM_front_tilt ;;
    front) echo CAM_front ;;
    orbit) echo CAM_orbit ;;
  esac
}

what=${1:-all}
case "$what" in
  all)
    render front_tilt CAM_front_tilt 1 30
    render front CAM_front 1 30
    render orbit CAM_orbit 1 90
    scripts/encode.sh
    ;;
  encode)
    scripts/encode.sh
    ;;
  front_tilt|front|orbit)
    if [ "$what" = orbit ]; then def_end=90; else def_end=30; fi
    render "$what" "$(cam_of "$what")" "${2:-1}" "${3:-$def_end}"
    ;;
  *)
    echo "usage: $0 [all|encode|front_tilt|front|orbit [start end]]" >&2
    exit 2
    ;;
esac
