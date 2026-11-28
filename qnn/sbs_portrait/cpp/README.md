# sbs_portrait 的 C++ / CUDA 版（学习用）

默认生产流水线（`pipeline.py` 的 `mode="opt"`、`official` 尺寸策略、`exact` profiles）的 C++ 实现，不依赖 torch。engine 仍由 Python 构建，这里只加载 `.plan`。支持画中画（`frame` / `icon`），不支持 `aligned` / `native`、`ref` 模式和 `--interleave`。

## 构建与运行

```
./build_deps.sh                                   # cmake、OpenCV 5.1.0、本工程（装在 ~/data/sbs/cpp）
python tools/export_assets.py --out ~/data/sbs/cpp/assets   # 模型常量、plan 路径、测试图标
~/data/sbs/cpp/build/sbs_portrait --assets ~/data/sbs/cpp/assets/assets.txt \
    --frames DIR --bg BG.jpg --out OUT [--kernel cuda|tex_point|tex_linear] [--pip frame|icon|png]
    [--sets DIR --sizes WxH,...] [--profile] [--op-times] [--dump] [--max-frames 150]
```

## 代码对照（建议按此顺序读）

| 文件 | 内容 | 对应 Python |
|---|---|---|
| `src/gpu.hpp` | device / pinned buffer、stream、event 的 RAII | torch 分配器与 Stream、`gpu_post.Pinned` |
| `kernels/post.cu` | warp、box（三种访存方式）；validity、膨胀、腐蚀、分位数、guided、EMA、fusion | `gpu_post.py`、`warp_box.py` |
| `kernels/resize.cu` | INTER_AREA 缩放 + 归一化 | `area_cuda.py`（生产用 `area_triton.py`） |
| `kernels/person.cu` | query 权重、person 融合 kernel、panoptic 规则 | `seg._person_fused`、`person_cuda.py` |
| `kernels/compose.cu` | 合成 + 画中画 | `pip_compose.py` |
| `src/trt_chain.cpp` | TensorRT C++ API：profile 选择、单 arena 三段链 | `trt_common.TRTChain` |
| `src/segmenter.cpp` | tap 权重、分割 | `seg.py` |
| `src/geometry.hpp` | SGBM、DIS、视差范围、线程池 | `temporal_stereo.py` |
| `src/pipeline.cpp` | 背景估计、逐帧编排、三条 stream、几何自适应并行 | `pipeline.py` |
| `src/main.cpp` | 解码预取、统计、dump | `run_pipeline.py`、`run_multires.py` |

## `--kernel`：访存方式

| 算子 | `cuda` | `tex_point` | `tex_linear` |
|---|---|---|---|
| 缩放 | global | 1-D 纹理读源字节 | BGRA 副本 + 硬件双线性，合并相邻 tap |
| person | global + shared 计数 | 1-D 半精度纹理 | 每帧拷到 layered array + 硬件双线性 |
| warp / box | global | 1-D 纹理 | pitch2D 硬件双线性 |
| 合成 / 画中画 | global | alpha 与源图走 1-D 纹理 | 硬件双线性（alpha 需 pitched 副本） |

逐点算子和归约（soft、EMA、fill、分位数、validity 等）只有一种实现：每个输入按线程自己的下标读一次，纹理通路没有可利用的东西。

硬件滤波权重是 8-bit 定点，`tex_linear` 的结果与 OpenCV / torch 不同（误差见 `../results/mr/cpp/LOG.md`）。

## 校验与测量

```
python tools/dump_ref.py --frames DIR --bg BG.jpg --out REF --n 6      # Python 的逐阶段张量
sbs_portrait ... --stage-dir CPP --stage-frames 6                      # C++ 的同名张量
python tools/compare.py REF CPP
./bench.sh [latency] [profile] [multires] [quality]                    # 与 Python 版对比
```

`--stage-dir` 的运行耗时不可用（在计时区间内下载张量）。

## 结果上屏

`--render gl|vulkan`（CUDA → OpenGL / Vulkan 零拷贝）及 `render_bench`：实现在 `render/`，接口在 `src/sink.hpp`，用法见 `../render/README.md`。
先跑 `../render/setup_deps.sh`；缺这些依赖时构建会自动跳过渲染部分。
