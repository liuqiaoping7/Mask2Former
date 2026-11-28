#!/usr/bin/env bash
# Cost of presenting the result: the 1080p pipeline without rendering vs with OpenGL / Vulkan, zero
# copy and host copy (C++ and Python, interleaved rounds), and the render path alone (render_bench,
# vsync off). Needs a desktop session (DISPLAY / XAUTHORITY).
#   render/bench.sh [pipeline] [path]      (default both; JSON in $OUT)
#   env: DISPLAY XAUTHORITY PY SBS_DATA BIN_DIR ASSETS OUT ROUNDS
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${SBS_DATA:-$HOME/data/sbs}
PY=${PY:-$HOME/anaconda3/envs/m2f/bin/python}
export PATH=$(dirname "$PY"):$PATH PYTHONDONTWRITEBYTECODE=1
BIN_DIR=${BIN_DIR:-$DATA/cpp/build}
ASSETS=${ASSETS:-$DATA/cpp/assets/assets.txt}
OUT=${OUT:-$DATA/render/bench}
ROUNDS=${ROUNDS:-3}
export LD_LIBRARY_PATH=$DATA/render/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p "$OUT"
awake() { gnome-session-inhibit --inhibit idle "$@"; }  # no screen blanking during a run

pipeline() {
  for r in $(seq "$ROUNDS"); do
    for cfg in none gl:zerocopy gl:copy vulkan:zerocopy vulkan:copy; do
      api=${cfg%:*}; present=${cfg#*:}
      flags=(); [ "$cfg" = none ] || flags=(--render "$api" --present "$present")
      for lang in cpp python; do
        d=$OUT/pipe_${lang}_${cfg/:/_}_$r
        if [ $lang = cpp ]; then
          awake "$BIN_DIR/sbs_portrait" --assets "$ASSETS" --bg "$DATA/bg_0045/0001.jpg" --frames "$DATA/portrait_0121" \
            --out "$d" "${flags[@]}" > "$d.log" 2>&1 || { tail -5 "$d.log"; exit 1; }
        else
          awake "$PY" run_pipeline.py --frames "$DATA/portrait_0121" --bg "$DATA/bg_0045/0001.jpg" --out "$d" \
            --no-preview "${flags[@]}" > "$d.log" 2>&1 || { tail -5 "$d.log"; exit 1; }
        fi
      done
    done
    echo "pipeline round $r done"
  done
}
path() {
  for r in $(seq "$ROUNDS"); do
    for size in 1920x1080 640x480; do
      for api in gl vulkan; do for present in zerocopy copy; do
        awake "$BIN_DIR/render_bench" --api $api --present $present --size $size 2>/dev/null |
          tail -1 > "$OUT/path_cpp_${api}_${present}_${size}_$r.json"
        awake "$PY" -m render.render_bench --api $api --present $present --size $size |
          tail -1 > "$OUT/path_python_${api}_${present}_${size}_$r.json"
      done; done
    done
    echo "path round $r done"
  done
}
[ $# -gt 0 ] || set -- pipeline path
for step in "$@"; do $step; done
"$PY" render/bench_summary.py "$OUT"
echo BENCH_DONE
