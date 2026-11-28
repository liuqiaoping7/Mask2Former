#!/usr/bin/env bash
# Python (production defaults) vs C++ (cuda / tex_point / tex_linear), preview encoding off on both.
#   ./bench.sh [latency] [profile] [multires] [quality]     (default: all; results in $OUT)
#   env: PY, SBS_DATA (~/data/sbs), BIN, ASSETS, OUT
set -euo pipefail
cd "$(dirname "$0")"
SBS=$(cd .. && pwd)
DATA=${SBS_DATA:-$HOME/data/sbs}
PY=${PY:-$HOME/anaconda3/envs/m2f/bin/python}
export PATH=$(dirname "$PY"):$PATH
BIN=${BIN:-$DATA/cpp/build/sbs_portrait}
ASSETS=${ASSETS:-$DATA/cpp/assets/assets.txt}
OUT=${OUT:-$DATA/cpp/bench}
BG=$DATA/bg_0045/0001.jpg
CLIP=$DATA/portrait_0121
SIZES=$($PY -c "import json;print(','.join(json.load(open('$DATA/mr/frames/sizes.json'))))")
KERNELS="cuda tex_point tex_linear"
mkdir -p "$OUT"

py() { (cd "$SBS" && "$PY" "$@"); }
cpp() { "$BIN" --assets "$ASSETS" --bg "$BG" "$@"; }
# run "$name" and record the launch time (startup = first_frame_unix - launch)
launch() { local d=$1; shift; mkdir -p "$d"; date +%s.%N > "$d/launch"; "$@" > "$d/log" 2>&1 || { tail -20 "$d/log"; exit 1; }; }

latency() {  # 1080p, 3 interleaved rounds x 150 frames
  for r in 1 2 3; do
    launch "$OUT/lat_python_$r" py run_pipeline.py --frames $CLIP --bg $BG --out "$OUT/lat_python_$r" --no-preview
    for k in $KERNELS; do launch "$OUT/lat_cpp_${k}_$r" cpp --frames $CLIP --kernel $k --out "$OUT/lat_cpp_${k}_$r"; done
    echo "latency round $r done"
  done
}
profile() {  # per-module breakdown (device sync around GPU modules)
  py run_pipeline.py --frames $CLIP --bg $BG --out "$OUT/prof_python" --no-preview --profile --max-frames 60 > "$OUT/prof_python.log" 2>&1
  for k in $KERNELS; do cpp --frames $CLIP --kernel $k --out "$OUT/prof_cpp_$k" --profile --max-frames 60 > "$OUT/prof_cpp_$k.log" 2>&1; done
  for p in frame icon; do
    py run_pipeline.py --frames $CLIP --bg $BG --out "$OUT/pip_${p}_python" --no-preview --pip $p > "$OUT/pip_${p}_python.log" 2>&1
    for k in $KERNELS; do cpp --frames $CLIP --kernel $k --pip $p --out "$OUT/pip_${p}_cpp_$k" > "$OUT/pip_${p}_cpp_$k.log" 2>&1; done
  done
  echo "profile done"
}
multires() {  # all sizes in one process, 150 frames, alpha dumps for the quality check
  py run_multires.py --sets $DATA/mr/frames --bg $BG --out "$OUT/mr_python" --no-preview --dump > "$OUT/mr_python.log" 2>&1
  for k in $KERNELS; do
    cpp --sets $DATA/mr/frames --sizes $SIZES --kernel $k --out "$OUT/mr_cpp_$k" --dump > "$OUT/mr_cpp_$k.log" 2>&1
    $PY tools/to_npz.py "$OUT"/mr_cpp_$k/*/
  done
  echo "multires done"
}
eval_run() {  # size run: eval.json against ref_official, eval_py.json against the Python run
  local d=$OUT/mr_$2/$1
  py eval_alpha.py --frames $DATA/mr/frames/$1 --run $d --ref $DATA/mr/ref_official/run/$1 > $d/eval_ref.log 2>&1
  mv $d/eval.json $d/eval_ref.json
  [ $2 = python ] && return
  py eval_alpha.py --frames $DATA/mr/frames/$1 --run $d --ref $OUT/mr_python/$1 > $d/eval_py.log 2>&1
  mv $d/eval.json $d/eval_py.json
}
quality() {
  for s in ${SIZES//,/ }; do
    for run in python cpp_cuda cpp_tex_point cpp_tex_linear; do eval_run $s $run & done
    wait
    echo "quality $s done"
  done
}
[ $# -gt 0 ] || set -- latency profile multires quality
for step in "$@"; do $step; done
echo BENCH_DONE
