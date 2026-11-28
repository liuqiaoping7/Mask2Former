#!/usr/bin/env bash
# Multi-resolution measurement (results in results/mr/<name>/), the multi-res counterpart of opt_step.sh.
#   ./mr_bench.sh <name> <mode> <size_policy> [profiles] [ref_name]
#   env: PY (python), SBS_DATA (default ~/data/sbs), SIZES (comma list, default all of sizes.json),
#        COMPOSE (compositing kernel, default triton), PERSON (cuda|triton), POST (triton|torch)
# 1) all sizes in one process, full 150 frames, no extra syncs: latency, NVML / RSS peaks, dumps
# 2) same with CUDA sync per GPU module, 40 frames: per-module breakdown per size
# 3) offline eval per size: 3D quality + error vs <ref_name> (same size)
set -euo pipefail
NAME=$1; MODE=$2; POLICY=$3; PROFILES=${4:-exact}; REF=${5:-}
PY=${PY:-python}; DATA=${SBS_DATA:-$HOME/data/sbs}
cd "$(dirname "$0")"
RUN=$DATA/mr/$NAME; mkdir -p "$RUN"
SETS=$DATA/mr/frames
ARGS="--sets $SETS --bg $DATA/bg_0045/0001.jpg --mode $MODE --size-policy $POLICY --profiles $PROFILES"
[ -n "${SIZES:-}" ] && ARGS="$ARGS --sizes $SIZES"
[ -n "${COMPOSE:-}" ] && ARGS="$ARGS --compose-kernel $COMPOSE"
[ -n "${PERSON:-}" ] && ARGS="$ARGS --person-kernel $PERSON"
[ -n "${POST:-}" ] && ARGS="$ARGS --post-kernels $POST"
$PY run_multires.py $ARGS --out "$RUN/run" --dump > "$RUN/run.log" 2>&1 || { tail -20 "$RUN/run.log"; exit 1; }
if [ "$MODE" = opt ]; then
  $PY run_multires.py $ARGS --out "$RUN/prof" --profile --max-frames 40 --key-frames "" > "$RUN/prof.log" 2>&1 || { tail -20 "$RUN/prof.log"; exit 1; }
fi
R=results/mr/$NAME; mkdir -p "$R"
cp "$RUN/run/summary.json" "$R/summary.json"
[ -f "$RUN/prof/summary.json" ] && cp "$RUN/prof/summary.json" "$R/summary_profile.json"
for d in "$RUN"/run/*/; do
  s=$(basename "$d")
  REFARG=""; [ -n "$REF" ] && REFARG="--ref $DATA/mr/$REF/run/$s"
  $PY eval_alpha.py --frames "$SETS/$s" --run "$d" $REFARG > "$d/eval.log" 2>&1 &
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 1; done
done
wait
for d in "$RUN"/run/*/; do
  s=$(basename "$d"); mkdir -p "$R/$s"
  cp "$d/stats.json" "$d/eval.json" "$R/$s/"
  [ -f "$RUN/prof/$s/stats.json" ] && cp "$RUN/prof/$s/stats.json" "$R/$s/stats_profile.json"
done
echo "MR_DONE $NAME"
