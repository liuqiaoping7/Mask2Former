#!/usr/bin/env bash
# Measure one optimization step of the production pipeline and store results in results/opt/<name>/.
#   ./opt_step.sh <name> [ref_name]     env: PY (python), SBS_DATA (default ~/data/sbs)
# 1) full run, no extra syncs: end-to-end latency, GPU memory (NVML), RSS, dumps for eval
# 2) profile run (CUDA sync per GPU module, 40 frames): per-module breakdown
# 3) offline eval: 3D quality metrics + error vs the reference run (default 00_baseline)
set -euo pipefail
NAME=$1; REF=${2:-00_baseline}
PY=${PY:-python}; DATA=${SBS_DATA:-$HOME/data/sbs}
cd "$(dirname "$0")"
RUN=$DATA/opt/$NAME; mkdir -p "$RUN"
ARGS="--frames $DATA/portrait_0121 --bg $DATA/bg_0045/0001.jpg"
$PY run_pipeline.py $ARGS --out "$RUN/run" --dump > "$RUN/run.log" 2>&1 || { tail -20 "$RUN/run.log"; exit 1; }
$PY run_pipeline.py $ARGS --out "$RUN/prof" --profile --max-frames 40 > "$RUN/prof.log" 2>&1 || { tail -20 "$RUN/prof.log"; exit 1; }
REFARG=""; [ "$NAME" != "$REF" ] && REFARG="--ref $DATA/opt/$REF/run"
$PY eval_alpha.py --frames $DATA/portrait_0121 --run "$RUN/run" $REFARG > "$RUN/eval.log" 2>&1 || { tail -20 "$RUN/eval.log"; exit 1; }
R=results/opt/$NAME; mkdir -p "$R"
cp "$RUN/run/stats.json" "$R/stats.json"; cp "$RUN/prof/stats.json" "$R/stats_profile.json"; cp "$RUN/run/eval.json" "$R/eval.json"
$PY - "$R" <<'PY'
import json, sys
R = sys.argv[1]
s, p, e = (json.load(open(f"{R}/{f}.json")) for f in ("stats", "stats_profile", "eval"))
print(f"e2e median {s['e2e_ms']['median']:.1f} ms (p95 {s['e2e_ms']['p95']:.1f}, {s['fps']:.2f} fps) | "
      f"GPU peak {s['gpu_mem_peak_mb_nvml']:.0f} MB (torch alloc {s['torch_max_allocated_mb']:.0f}) | RSS peak {s['rss_peak_mb']:.0f} MB")
print("modules (sync):", {k: round(v, 1) for k, v in p["modules_ms"].items()})
print("quality:", {k: round(v, 4) for k, v in e["quality"].items()})
if "error_vs_ref" in e:
    print("error vs ref:", e["error_vs_ref"])
PY
