# sbs_portrait 优化日志

测量：`./opt_step.sh <编号_名称> [参考]`，结果在 `results/opt/<编号_名称>/`（stats.json 端到端 / 显存 / RSS，stats_profile.json 逐模块，eval.json 质量与误差，均含 commit）。
测量波动约 ±5 ms。精度容差：alpha MAE ≤ 0.01、IoU 均值 ≥ 0.99、3D 指标变化 ≤ ±5%（相对 00_baseline）。
环境：RTX 4060 8 GB（驱动 595.58.03）、i5-12400F、CUDA 12.8、torch 2.10、TensorRT 10.13.3.9。

## 第一轮（0～11，详见 QNN/docs/Mask2Former-sbs-portrait-opt.md）

| # | 层次 | 改动 | 端到端 ms | 显存峰值 MB | 结论 |
| - | ---- | ---- | --------- | ----------- | ---- |
| 0 | 系统 | 生产流水线 + 测量框架 | 1264.7 | 7300 | 基线 |
| 1～11 | 各层 | 见文档 | → 213.7 | → 2140 | 保留；引擎（backbone TRT、decoder torch.compile）与精度（FP16）冻结 |
| ✗ | 并行 | 双眼 batch=2 | +49 | +1.9 GB | 回退 |

第一轮结束候选池：pixel decoder 转 TRT（MSDA plugin）、整段 CUDA Graph、跨帧流水线（只提吞吐）、降精度 / 降分辨率（突破约束）、NVOF / NVDEC（释放 CPU）、RSS。

## 第二轮

起点：复用 11_compile_decoders 的数据（代码到 361a88d 只加 NVTX 与文档，环境未变）。
补测（只补缺的）：`prof/seg_breakdown.py` → `results/r2/seg_breakdown/`：每眼 backbone 43.0 / pixel decoder 34.9 / transformer decoder 8.75 ms；pixel decoder 内 FP32 MSDA im2col 22%（1.3 ms/层，value 仅 21 MB），~15 ms 为 autocast 下 FP32 残差流带来的 dtype 转换与逐元素 kernel；估算计算量 ~270 GFLOP（FP16 理想 ~9 ms）。

| # | 层次 | 改动 | 端到端 ms | 显存峰值 MB | RSS MB | alpha MAE / IoU 均值 / 最差 | ghost | 结论 | 原因 |
| - | ---- | ---- | --------- | ----------- | ------ | --------------------------- | ----- | ---- | ---- |
| 12 | 引擎 | pixel decoder → TensorRT FP16（内置 MultiscaleDeformableAttnPlugin_TRT），位置编码常量化，多尺度输出 NHWC | 177.3（−36.4） | 1950 | 2643 | 3.5e-4 / 0.9981 / 0.9872 | 0.0122 | 保留，冻结 pixel decoder 引擎 | 每眼 pixel decoder 34.9→18.1 ms |
| 13 | 系统 / kernel | 模型输入预处理移到 GPU：OpenCV INTER_AREA 权重表 + cvRound + RGB + 归一化融合为一个 Triton kernel，从已上传的整帧直接读 | 168.5（−8.8） | 1950 | 2640 | 3.5e-4 / 0.9981 / 0.9871 | 0.0122 | 保留 | CPU 关键路径 7.4→0.5 ms；GPU 0.09 ms/眼；与 cv2 逐像素 99.98% 相同、最大差 1 |
| 14 | 数据流 | 几何结果（灰度、视差、4 个光流，20.8 MB/帧）在几何线程算完即用独立 copy stream 上传，与 GPU 分割重叠 | 166.4（−2.1） | 1972 | 2636 | 3.5e-4 / 0.9981 / 0.9872 | 0.0122 | 保留 | gpu.post 13.0→7.8 ms；原同步路径上传 4.6 ms，部分本已与计算重叠，端到端 −2.1 ms（p95−中位 0.3 ms，>波动） |
| 15 | 算法 | 后处理：31×31 椭圆膨胀改为行内最近前景距离 + 按行半宽判定（与卷积逐像素一致），9×9 box 改可分离并批量化 | 163.0（−3.4） | 1970 | 2597 | 3.5e-4 / 0.9981 / 0.9872 | 0.0122 | 保留 | gpu.post 7.8→4.5 ms；膨胀 2.15→0.40 ms/次（原为 961 抽头 TF32 卷积，占后处理 52%） |

### 第 12 项过程记录（上限试验 `prof/pd_spike.py` → `results/r2/pd_spike*`）
- MSDA 插件单测：与上游 CUDA 核逐位一致（max diff 0）。
- 首版 engine 误差 6%～12%（FP32 / FP16 相同）→ 二分定位到 PositionEmbeddingSine 在 TRT 中算错（~40%），改为按形状预计算常量后 FP32 engine 误差 1.4e-4。
- FP16 engine 输出误差 0.6%～1.5%（torch FP16 路径 0.05%）。尝试：采样相关层 + 插件固定 FP32（26 ms，误差不变）、非 GEMM 层全 FP32（24.7 ms，0.3%～0.9%）、MatMul FP32 输出（误差不变）→ 误差不来自插件 / 采样坐标 / 累加；端到端指标在容差内，选纯 FP16（18.1 ms）。
- 接入后只省 16 ms：transformer decoder 8.75→23 ms，原因是 TRT 输出 NCHW 连续，而 torch 路径的多尺度特征是 channels-last 步长；engine 改为 NHWC 输出后恢复 8.8 ms。
- 测试脚本在默认 stream 与 seg.stream 交替调用同一 TRT context 时出现 illegal memory access（CUDA_LAUNCH_BLOCKING 下消失）：TRT 封装加跨 stream 顺序保护。
### 第 13 项过程记录（`prof/check_preprocess_gpu.py`）
- 预取线程里做下一帧预处理只缩短循环周期，不降低单帧延迟，未采用；改为 GPU 预处理（原帧本来就上传给合成用）。
- gather + sum 实现 2.1 ms/眼（K 倍中间张量），GPU 是瓶颈时净收益只有 ~3 ms；融合 Triton kernel 0.09 ms/眼。
- seg.model 同时 −5 ms：不再从 pageable 内存上传预处理后的 uint8 输入。

### 其他
- 静默回退：Inductor 缓存实际写在 /tmp/torchinductor_qiao，而非 weights/inductor_cache（待修，见候选 F）。
