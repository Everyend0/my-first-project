#!/usr/bin/env bash
# Encode rendered PNG frames into 3840x2160, 30 fps H.264 MP4 files.
set -euo pipefail
cd "$(dirname "$0")/.."
FR=build/frames
OUT=renders
mkdir -p "$OUT"

enc() {  # name loop(0/1)
  local name=$1 loop=$2 n
  n=$(ls "$FR/$name"/*.png | wc -l)
  local loopopt=()
  [ "$loop" = 1 ] && loopopt=(-loop 1)
  ffmpeg -hide_banner -loglevel error -y \
    -framerate 30 -start_number 1 "${loopopt[@]}" -i "$FR/$name/%04d.png" \
    -frames:v 90 \
    -vf "scale=3840:2160:flags=lanczos:out_color_matrix=bt709:out_range=tv,format=yuv420p" \
    -c:v libx264 -preset slow -crf 14 -tune animation \
    -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
    -r 30 -movflags +faststart \
    "$OUT/iris_${name}_4k30.mp4"
  echo "encoded $OUT/iris_${name}_4k30.mp4 from $n frames"
}

enc front_tilt 1
enc front 1
enc orbit 0
