#!/usr/bin/env bash
# Encode composited PNG frames into 3840x2160, 30 fps, exactly 90-frame
# (3.0 s) H.264 MP4 files, and check the result.
set -euo pipefail
cd "$(dirname "$0")/.."
FR=build/frames
OUT=renders
mkdir -p "$OUT"

need_frames() {  # dir count: every %04d.png 0001..count must exist
  local dir=$1 count=$2 i
  for ((i = 1; i <= count; i++)); do
    if [ ! -s "$(printf '%s/%04d.png' "$dir" "$i")" ]; then
      echo "missing frame $(printf '%04d' "$i") in $dir" >&2
      exit 1
    fi
  done
}

enc() {  # name frames_rendered (30 = 1 s loop played 3 times, 90 = all frames)
  local name=$1 rendered=$2 out="$OUT/iris_${1}_4k30.mp4"
  need_frames "$FR/$name/final" "$rendered"
  local args=(-hide_banner -loglevel error -y -framerate 30 -start_number 1)
  if [ "$rendered" -lt 90 ]; then
    args+=(-loop 1)                      # image2: repeat the 1 s loop
  fi
  args+=(-i "$FR/$name/final/%04d.png" -frames:v 90
    -vf "scale=3840:2160:flags=lanczos:out_color_matrix=bt709:out_range=tv,format=yuv420p"
    -c:v libx264 -preset slow -crf 14 -tune animation -level:v 5.1
    -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv
    -r 30 -movflags +faststart "$out")
  ffmpeg "${args[@]}"
  # verify: 3840x2160, 30 fps, 90 frames
  local probe
  probe=$(ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=width,height,r_frame_rate,nb_read_frames -of csv=p=0 "$out")
  if [ "$probe" != "3840,2160,30/1,90" ]; then
    echo "unexpected stream in $out: $probe" >&2
    exit 1
  fi
  echo "encoded $out ($probe)"
}

enc front_tilt 30
enc front 30
enc orbit 90
