#!/usr/bin/env bash
# Full-pipeline profiling of sbs_portrait_bench.py (see QNN docs/Mask2Former-session.md section 6).
# Needs: NVreg_RestrictProfilingToAdminUsers=0 (ncu), kernel.perf_event_paranoid<=1 (perf, nsys CPU sampling).
#   ./run_prof.sh <out_dir>      env: PY (python), NCU, SBS_DATA
set -euo pipefail
OUT=${1:-prof_out}; mkdir -p "$OUT"; cd "$(dirname "$0")"; OUT=$(realpath "$OUT")
PY=${PY:-python}; NCU=${NCU:-ncu}

# 1) per-module wall time (sync around GPU modules) + torch profiler stage FLOPs
$PY prof_pipeline.py > "$OUT/pipeline.json" 2> "$OUT/pipeline.err"

# 2) nsys timeline without extra syncs (NVTX ranges per module) + CPU sampling
PROF_N=10 PROF_NOSYNC=1 PROF_SKIP_B=1 nsys profile -t cuda,nvtx,osrt -s cpu --cpuctxsw=process-tree \
  -o "$OUT/pipe" -f true $PY prof_pipeline.py > "$OUT/pipe_nosync.json" 2> "$OUT/pipe_nosync.err"
nsys export -t sqlite -f true -o "$OUT/pipe.sqlite" "$OUT/pipe.nsys-rep" > /dev/null
python3 nsys_analyze.py "$OUT/pipe.sqlite" > "$OUT/nsys_summary.txt"

# 3) ncu SOL of every kernel in one eye's forward, grouped by model stage (NVTX)
MET=gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
INC=""; for S in S1_backbone S2_pixel_decoder S3_transformer_decoder S4_mask_upsample_to_input \
  S5_upsample_crop_1080p S6_semantic_inference S7_panoptic_inference S8_person_prob; do INC="$INC --nvtx-include $S/"; done
$NCU --replay-mode application --clock-control none --nvtx $INC --metrics $MET --csv --page raw $PY ncu_stage.py > "$OUT/ncu_all.csv" 2> "$OUT/ncu_all.err"
python3 ncu_parse.py "$OUT/ncu_all.csv" > "$OUT/ncu_summary.md"

# 4) perf top-down (TmaL1/L2) per CPU module, counting only the timed loop
rm -f "$OUT/ctl" "$OUT/ack"; mkfifo "$OUT/ctl" "$OUT/ack"
echo "module;ms_per_iter;metrics..." > "$OUT/cpu_topdown.csv"
for m in imread pil_resize sgbm_LR dis_flow_x4 flow_validity_x2 disp_fill guided_x2 ema_x2 compose_blend_x2 compose_clip_u8; do
  out=$(perf stat -M TmaL1,TmaL2 --metric-only -x ";" -D -1 --control "fifo:$OUT/ctl,$OUT/ack" -- \
        $PY cpu_modules.py $m 3 "$OUT/ctl" "$OUT/ack" 2>&1)
  ms=$(echo "$out" | sed -n "s/.*ms_per_iter=\([0-9.]*\).*/\1/p")
  [ "$(wc -l < "$OUT/cpu_topdown.csv")" -gt 1 ] || echo "module;ms_per_iter;$(echo "$out" | grep '^tma_info' | head -1)" > "$OUT/cpu_topdown.csv"
  echo "$m;$ms;$(echo "$out" | grep -A1 '^tma_info' | tail -1)" >> "$OUT/cpu_topdown.csv"
done
rm -f "$OUT/ctl" "$OUT/ack"
echo "done: $OUT"
