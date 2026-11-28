set -euo pipefail
export PY=$HOME/anaconda3/envs/m2f/bin/python PATH=$HOME/anaconda3/envs/m2f/bin:$PATH
M=$HOME/Documents/github/QNN/Mask2Former/qnn; NEW=$M/sbs_portrait; OLD=$M/_head_tmp; D=$HOME/data/sbs; W=$D/mr/kern
rm -rf $W; mkdir -p $W
ARGS="--frames $D/portrait_0121 --bg $D/bg_0045/0001.jpg"
# 1) 1080p A/B latency, 3 interleaved rounds
for r in 1 2 3; do
  (cd $OLD && $PY run_pipeline.py $ARGS --out $W/lat_head_$r > $W/lat_head_$r.log 2>&1)
  for cfg in "cuda triton" "cuda torch" "triton triton" "triton torch"; do
    set -- $cfg
    (cd $NEW && $PY run_pipeline.py $ARGS --out $W/lat_${1}_${2}_$r --person-kernel $1 --post-kernels $2 > $W/lat_${1}_${2}_$r.log 2>&1)
  done
  echo ROUND $r
done
# 2) outputs: all 10 sizes, old code vs new default; 1080p clip too
BG="--sets $D/mr/frames --bg $D/bg_0045/0001.jpg --dump"
(cd $OLD && $PY run_multires.py $BG --out $W/mr_head > $W/mr_head.log 2>&1)
(cd $OLD && $PY run_pipeline.py $ARGS --out $W/dump_head --dump > $W/dump_head.log 2>&1)
(cd $NEW && $PY run_pipeline.py $ARGS --out $W/dump_new --dump > $W/dump_new.log 2>&1)
(cd $NEW && $PY run_pipeline.py $ARGS --out $W/prof_new --profile --max-frames 40 > $W/prof_new.log 2>&1)
(cd $OLD && $PY run_pipeline.py $ARGS --out $W/prof_head --profile --max-frames 40 > $W/prof_head.log 2>&1)
echo MR_NEW
(cd $NEW && ./mr_bench.sh opt_kernels opt official exact ref_official > $W/mr_bench.log 2>&1)
(cd $NEW && $PY eval_alpha.py --frames $D/portrait_0121 --run $W/dump_new --ref $D/opt/00_baseline/run > $W/eval_new.log 2>&1)
(cd $NEW && $PY eval_alpha.py --frames $D/portrait_0121 --run $W/dump_head --ref $D/opt/00_baseline/run > $W/eval_head.log 2>&1)
echo ALL_DONE
