# SBS 双目人像分割 + SBS 虚拟背景 benchmark

双目 SBS（左 | 右，每眼 VGA～1080p 任意分辨率）视频逐眼做人像分割，与 SBS 立体虚拟背景逐眼融合，并评估连续帧下 mask 的时域稳定性与左右眼一致性。完整分析与结论见 QNN 仓库 `docs/Mask2Former-session.md`。

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

- `pipeline.py`：生产流水线（分割 → 视差/光流 → soft → 引导滤波 → 时域 EMA → 立体融合 → 全分辨率合成 + 可选画中画），逐模块计时。
- `run_pipeline.py`：端到端延迟、逐模块耗时（`--profile` 时 GPU 模块前后同步）、NVML 显存峰值、RSS 峰值；`--dump` 保存逐帧 alpha 供离线评估。
- `eval_alpha.py`：离线计算 3D 质量指标，并与参考运行（基线）对比 alpha 误差和关键帧 PSNR。
- `opt_step.sh <name>`：一次优化的标准测量流程，结果写入 `results/opt/<name>/`。

### 优化后的流水线（默认开启）

| 模块 | 实现 | 开关 |
| ---- | ---- | ---- |
| 模型输入缩放 | GPU：OpenCV INTER_AREA 权重 + 归一化融合的 Triton kernel（`area_triton.py`），直接读已上传的整帧 | `pre=[seg.preprocess_gpu(eye)]` |
| backbone | TensorRT FP16 动态形状 engine，形状无关的 Swin 导出（`dyn_export.py`、`trt_backbone.py`） | `backbone="trt"`、`profiles` |
| pixel decoder | TensorRT FP16 engine，MSDA 用内置 `MultiscaleDeformableAttnPlugin_TRT`，闭式位置编码，多尺度特征 NHWC 输出（`trt_pixel_decoder.py`） | `pixel_decoder="trt"` |
| transformer decoder | 推理专用 forward 导出的 TensorRT FP16 engine，注意力 mask 为有限值加性 mask（`trt_decoder.py`） | `fast_decoder`、`decoder="trt"` |
| TensorRT 运行时 | 多 optimization profile（常见离散尺寸用精确 profile + ranged 兜底），共享激活内存、按最大形状分配的输出缓冲、懒创建执行上下文（`trt_common.py`） | `profiles="exact"` |
| person 后处理 | Triton 融合 kernel（`person_triton.py`），直接输出到工作网格 | `fused` |
| alpha 后处理 + 合成 | GPU（`gpu_post.py`：精确等价的快速椭圆膨胀、可分离 box），pinned 缓冲上传 / 回传 | `post="gpu"` |
| 视差 / 光流 | CPU 工作线程，与 GPU 分割并行；结果在工作线程内经独立 copy stream 上传；几何晚于 GPU 时自动拆成 4 路并行 | `geo_threads`、`geo_late_ms` |
| 合成 + 画中画 | Triton 融合合成 kernel + 画中画 kernel（`pip_compose.py`，另有 torch / CUDA / CUDA 纹理采样实现） | `compose_kernel`、`pip` |

首次运行在 `weights/` 下构建三个 engine（`exact` 集合 14 个 profile：约 5.5～11 分钟，主机内存峰值约 10 GB，plan 共 653 MB），之后启动约 7 s。engine 与 GPU 型号、TensorRT 版本绑定，不入库。`pixel_decoder="torch"` 路径的 `compile_decoders` 仍可用，其 Inductor 缓存写在 `weights/inductor_cache`。逐项优化的测量结果见 `results/opt/`（`rejected_*` 为已回退的尝试），过程与候选池见 `results/opt/LOG.md`。

## 多分辨率（每眼 VGA～1080p）

- 尺寸策略 `size_policy`：`official`（默认，短边 800 / 长边 ≤ 1333，与 detectron2 测试尺寸一致）、`aligned`（边长对齐 Swin 第三阶段窗口，满足精度容差，非 16:9 尺寸快 5%～14%）、`native`（小图不放大，VGA 2.2 倍提速，边界质量下降）。
- 后处理在约 960x540 像素的工作网格上进行，像素尺寸的算子按工作网格高度缩放；1080p 时与原实现完全相同。
- `make_multires.py`：由 1080p 片段生成测试分辨率（6 种常见 + 4 种随机）；`run_multires.py`：一个进程依次 / 交替处理多种分辨率；`mr_bench.sh <name> <mode> <size_policy>`：测量 + 逐模块 + 每种分辨率离线评估，结果在 `results/mr/<name>/`；`prof/mr_tables.py` 生成汇总表。
- 画中画：`--pip frame`（本地 SBS 原图）或 `--pip icon` / `--pip <SBS RGBA png>`，每眼缩放到 H/8 × W/8 放在右上角，深度在人物之前。
- 试验与结论见 `results/mr/LOG.md`，完整分析见 QNN 仓库 `docs/Mask2Former-sbs-portrait-opt.md` 第十一节。
