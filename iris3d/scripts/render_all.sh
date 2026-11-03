#!/usr/bin/env bash
# Render the 4K frames and encode the three 4K 30 fps videos.
#
#   scripts/render_all.sh                 # everything (~5 h on 4 CPU cores)
#   scripts/render_all.sh orbit 1 45      # one camera, a frame range (render + composite)
#   scripts/render_all.sh stills          # 4K stills for the README
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
# composite_iris.py. Finished frames are skipped on a re-run, so after
# changing the scene delete build/frames first (stills are always re-rendered).
set -euo pipefail
cd "$(dirname "$0")/.."
BL=scripts/blender.sh
FR=build/frames

blender_pass() {  # camera start end pass outdir [extra args...]
  local cam=$1 s=$2 e=$3 pass=$4 out=$5
  shift 5
  # --python-exit-code makes Blender fail on a Python error; the grep is
  # grouped so it cannot hide Blender's exit status from pipefail
  "$BL" -b model/iris.blend --python-exit-code 1 --python scripts/render_iris.py -- \
      --camera "$cam" --start "$s" --end "$e" --pass "$pass" --out "$out" "$@" 2>&1 \
      | { grep --line-buffered -E "^(FRAME|DONE)|Error" || true; }
}

render() {  # name camera start end
  local name=$1 cam=$2 s=$3 e=$4
  blender_pass "$cam" "$s" "$e" shadow "$FR/$name/shadow" --samples 32
  blender_pass "$cam" "$s" "$e" char "$FR/$name/char"
  python3 scripts/composite_iris.py "$FR/$name/char" "$FR/$name/shadow" "$FR/$name/final"
}

stills() {
  rm -rf build/stills            # 4 short renders; never reuse stale stills
  mkdir -p renders
  # rest pose (the drawing's shape), straight on and from 15 deg above
  for cam in CAM_front CAM_front_tilt; do
    local d=build/stills/$cam
    blender_pass "$cam" 1 1 shadow "$d/shadow" --samples 32 --rest
    blender_pass "$cam" 1 1 char "$d/char" --rest
    python3 scripts/composite_iris.py "$d/char" "$d/shadow" "$d/final"
  done
  cp build/stills/CAM_front/final/0001.png renders/still_front_rest.png
  cp build/stills/CAM_front_tilt/final/0001.png renders/still_front_tilt_rest.png
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
    stills
    scripts/encode.sh
    ;;
  stills)
    stills
    ;;
  encode)
    for name in front_tilt front orbit; do
      python3 scripts/composite_iris.py "$FR/$name/char" "$FR/$name/shadow" "$FR/$name/final"
    done
    scripts/encode.sh
    ;;
  front_tilt|front|orbit)
    if [ "$what" = orbit ]; then def_end=90; else def_end=30; fi
    render "$what" "$(cam_of "$what")" "${2:-1}" "${3:-$def_end}"
    ;;
  *)
    echo "usage: $0 [all|stills|encode|front_tilt|front|orbit [start end]]" >&2
    exit 2
    ;;
esac
