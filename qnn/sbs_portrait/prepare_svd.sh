#!/usr/bin/env bash
# Download SVD (Spatial Video Dataset, CC BY-NC 4.0, https://cd-athena.github.io/SVD/)
# iPhone 16 Pro MV-HEVC stereo clips (1920x1080@30, baseline 19.2 mm) and decode
# them into full side-by-side frames (3840x1080, left | right).
#
# Needs FFmpeg >= 7.1 (MV-HEVC multilayer decoding), e.g. the static binary from
# `pip install imageio-ffmpeg`.  Set FFMPEG=/path/to/ffmpeg if not on PATH.
#
#   prepare_svd.sh download <data_dir> <id...>        # e.g. 0001 0002
#   prepare_svd.sh thumbs   <data_dir> <out_dir>      # frame 0 SBS of every clip
#   prepare_svd.sh frames   <data_dir> <id> <out_dir> [n_frames]
set -euo pipefail
FFMPEG=${FFMPEG:-ffmpeg}
URL=https://ftp.itec.aau.at/datasets/SVD/iPhone
# In SVD iPhone clips view 0 is tagged "right" and view 1 "left".
SBS='[0:v:view:1][0:v:view:0]hstack[v]'

cmd=$1; data=$2; shift 2
mkdir -p "$data"
case $cmd in
download)
  for id in "$@"; do
    [ -s "$data/$id.mov" ] || curl -sSf -o "$data/$id.mov" "$URL/$id.mov"
  done ;;
thumbs)
  out=$1; mkdir -p "$out"
  for f in "$data"/*.mov; do
    id=$(basename "$f" .mov)
    [ -s "$out/$id.jpg" ] || "$FFMPEG" -hide_banner -loglevel error -y -i "$f" \
      -filter_complex "$SBS" -map '[v]' -frames:v 1 -q:v 3 "$out/$id.jpg"
  done ;;
frames)
  id=$1; out=$2; n=${3:-150}; mkdir -p "$out"
  "$FFMPEG" -hide_banner -loglevel error -y -i "$data/$id.mov" \
    -filter_complex "$SBS" -map '[v]' -frames:v "$n" -q:v 2 "$out/%04d.jpg"
  ls "$out" | wc -l ;;
*) echo "unknown command $cmd"; exit 1 ;;
esac
