# SBS 双目人像分割 + SBS 虚拟背景 benchmark

双目 SBS（左 | 右，每眼 1080p）视频逐眼做人像分割，与 SBS 立体虚拟背景逐眼融合，并评估连续帧下 mask 的时域稳定性与左右眼一致性。完整分析与结论见 QNN 仓库 `docs/Mask2Former-session.md`。

## 依赖

- detectron2、本仓库 Mask2Former（需编译 `mask2former/modeling/pixel_decoder/ops`）、opencv-python。
- FFmpeg ≥ 7.1（解码 iPhone 空间视频 MV-HEVC），例如 `pip install imageio-ffmpeg` 自带的静态版。
- 权重放到仓库根目录 `weights/`（默认 Swin-L COCO panoptic）：
  ```bash
  mkdir -p weights && cd weights
  curl -LO https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl
  ```

## 数据

[SVD: Spatial Video Dataset](https://cd-athena.github.io/SVD/)（CC BY-NC 4.0）iPhone 16 Pro 空间视频，每眼 1920×1080@30，基线 19.27 mm。人像 `0121`，背景 `0045`。

```bash
export FFMPEG=/path/to/ffmpeg
./prepare_svd.sh download ~/data/svd 0121 0045
./prepare_svd.sh frames ~/data/svd 0121 ~/data/sbs/portrait_0121 150
./prepare_svd.sh frames ~/data/svd 0045 ~/data/sbs/bg_0045 1
# 可选：对全部 150 段片段打分，挑选人像/背景片段
./prepare_svd.sh download ~/data/svd $(seq -f "%04g" 1 150)
./prepare_svd.sh thumbs ~/data/svd ~/data/svd_thumbs
python select_clips.py --thumbs ~/data/svd_thumbs --out ~/data/sbs/select
```

## 运行

```bash
python sbs_portrait_bench.py --frames ~/data/sbs/portrait_0121 --bg ~/data/sbs/bg_0045/0001.jpg \
    --cache ~/data/sbs/cache_swinl --out ~/data/sbs/run
```

- 输出：`metrics.md/json`（各配置指标与延迟）、`composite_*.mp4`、关键帧 SBS 合成图与 alpha、边界抖动热力图。
- `--cache` 缓存逐帧分割结果（含分割耗时），调后处理时不必重复推理。
- 默认逐眼推理（Swin-L 在 8 GB 显存上无法双眼 batch）；显存足够时加 `--batch-eyes`。
- 各配置含义见 `sbs_portrait_bench.py` 文件头；`results/` 为当前提交在 RTX 4060 上的实测结果。

## 生产流水线与优化测量

`sbs_portrait_bench.py` 每帧要算 6 个对比配置和全部一致性指标，用于算法对比；部署只走 final 一条路径：

- `pipeline.py`：生产流水线（分割 → 视差/光流 → soft → 引导滤波 → 时域 EMA → 立体融合 → 1080p 合成），逐模块计时。
- `run_pipeline.py`：端到端延迟、逐模块耗时（`--profile` 时 GPU 模块前后同步）、NVML 显存峰值、RSS 峰值；`--dump` 保存逐帧 alpha 供离线评估。
- `eval_alpha.py`：离线计算 3D 质量指标，并与参考运行（基线）对比 alpha 误差和关键帧 PSNR。
- `opt_step.sh <name>`：一次优化的标准测量流程，结果写入 `results/opt/<name>/`。
