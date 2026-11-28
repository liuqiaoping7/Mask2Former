# 多分辨率（第三轮）测量记录

环境：RTX 4060 8 GB、i5-12400F、PyTorch 2.10、TensorRT 10.13.3.9。数据：`make_multires.py` 生成的 10 种分辨率（150 帧，0121），背景 0045。容差：对同分辨率参考版（`mode="ref"`，PyTorch FP32 + CPU 后处理）alpha MAE ≤ 0.01、IoU 均值 ≥ 0.99、3D 指标 ±5%。

| 目录 / 文件 | 内容 | 结论 |
| ---- | ---- | ---- |
| `dyn_spike_eager.json`、`dyn_spike_fp32_*.json` | 形状无关 forward 对上游（eager）、动态 engine 对 PyTorch（严格 FP32），4 种尺寸 | backbone 逐位一致 / ≤1e-5；pixel decoder ≤2.6e-5 / ≤1.9e-4；decoder ≤6e-7 / ≤3e-6。默认 FP32 构建允许 TF32，decoder 误差 0.5%～1.6%，关掉后才能判断转换正确 |
| `ref_official/`、`ref_native/` | 参考版，各分辨率 | 866～1153 ms/帧 |
| `opt_official/`、`opt_native/` | multires 集合（p1080 + 3 个 ranged 桶），official / native 策略 | 全部尺寸满足容差；800 行输入慢（1396x842 179 ms）；native VGA 60.6 ms 但 IoU 0.9868 |
| `profile_spike_{static1080,wide,multires,exact}.json`、`profile_compare_*.json` | 每眼分割链耗时，四种 profile 方案 | 宽 profile 慢 3%～16%；精确 profile 比 ranged 快 4%～6%，1080p 与静态 engine 相同（64.0 vs 63.6 ms）；backbone 耗时与 Swin 窗口补齐后的 FLOPs 成正比 |
| `opt_final/` | exact 集合（14 个 profile），official 策略 | 默认配置的多分辨率结果 |
| `geo_spike_960x540.json`、`opt_native_geo/`、`opt_final_native/` | 几何并行：试验、总是并行、自适应 | 960x540 103.5 → 90.6 ms；不需要时并行会慢 1～3 ms，故按 GPU 空等时间（CUDA 事件）> 5 ms 自适应开启；676x504 在临界点被开启（+2.7 ms） |
| `opt_aligned/`、`opt_final_aligned/` | aligned 尺寸策略 | 非 16:9 尺寸 −5%～−14%，满足容差（IoU 均值 ≥ 0.994） |
| `interleave/` | 三种分辨率每帧交替 vs 单一分辨率 | 差 −1.6～+4.0 ms |
| `batch/batch_spike.json` | 所有尺寸双眼 batch=2 vs 两次 batch=1 | 全部更慢（+3.3%～+11%），激活显存 ×2～2.5；4060 上单眼已占满 24 个 SM |
| `pip/pip_spike.json`、`pip/crop_*.png`、`pip/icon_3d_sbs.png` | 合成 + 画中画的 6 种实现，10 种分辨率 | Triton（合成 kernel + 画中画 kernel）最快且逐位一致：1080p 0.24 ms（torch 3.89 ms）；纹理采样对 RGBA 图标省一半取样时间，对 3 通道帧要先转 RGBA，整体更慢 |
| `pip/e2e/` | 端到端：合成实现 × 画中画来源 | 画中画几乎零开销；Triton 合成每帧省 1～3 ms |
| `area/area_spike.json`、`area_spike.log` | 模型输入缩放 + 归一化：Triton vs CUDA global / 纹理点采样 / 纹理硬件双线性（`area_cuda.py`，`prof/area_spike.py`），10 种分辨率；基于 393279f | Triton 最快：每眼 kernel 0.043～0.081 ms，CUDA global 0.070～0.090，纹理点采样 0.075～0.094（比 CUDA global 慢 1%～8%），双线性（含 BGRA 转换）0.079～0.155；global 与点采样和 Triton 输出逐位一致，对 cv2 最大差 1。CUDA 对照组慢于 Triton 的原因未查明（`-prec-div=false` 无效，每线程 2 像素只快约 10%） |
| `area/pip_rerun.json` | 合成 + 画中画的纹理实现复测（`prof/pip_spike.py`，代码未改） | 结论与 `pip/` 一致 |
| `ops/ops_spike.json`、`ops_spike.log` | 其余采样算子，按 CUDA → 纹理 → Triton：warp、box（`warp_box.py`，基线为 torch）、person 融合（`person_cuda.py`，基线为 Triton），`prof/ops_spike.py`，真实输入，10 种分辨率；基于 393279f | warp / box 手写版比 torch 快 4～10 倍；纹理点采样与 CUDA 持平，硬件滤波多数更慢且有定点权重误差；person 的 CUDA 版比生产 Triton 版快（1080p 单眼 1.15 vs 1.53 ms，输出一致）：Triton 版每 query 每 block 做全局 atomic_add，临时改成每 block 部分和（输出逐位一致）在合成输入上 1.76 → 1.46 ms，同批 CUDA 1.23 ms。部分和改法已应用到 `person_triton.py`，真实输入上 1080p 单眼 1.53 → 1.33 ms，仍慢于 CUDA（1.17～1.20） |
| `kernels/`、`opt_kernels/` | 接入生产：person 用 CUDA kernel（`--person-kernel`），warp / box 用 Triton（`--post-kernels`）；对 393279f 原代码的 1080p 交错 A/B（3 轮）、输出对比与 10 种分辨率回归 | 1080p 144.5 → 142.2 ms（只换 person −0.9，只换 warp / box 约 −1.5），各分辨率 −1.0～−3.1 ms，显存峰值不变；hard mask 逐位相同，alpha MAE ≤ 9e-8（个别像素最大差 0.029，浮点舍入），3D 指标对 opt_final 变化 ≤ 0.001%。同一份原代码本次 144.5 ms，低于早先记录的 149.3 ms，故只比较同批数据 |
| `msda_plugin/` | 内置 MSDA 插件 vs 包装工程自带 CUDA kernel 的 IPluginV3 插件：单算子、整个 pixel decoder（3 种尺寸 × FP32 / FP16）、1080p 端到端 | FP32 下两者与上游 kernel 逐位一致；FP16 engine 中上游插件保持 FP32，pixel decoder 误差低 10%～18%（0.37% → 0.33%），端到端无差别（alpha 对 baseline 2.55e-4 vs 2.54e-4），每帧慢 8.7 ms；仅作对照 |
| `arena/` | 三段 TensorRT 的 scratch 显存共享：池按需增长 → 乒乓 arena → 段间特征 FP16；各段布局、workspace 上限试验、1080p 与多分辨率进程的显存 | 1080p 显存峰值 1458 → 1230 → 1148 → 1078 MB，延迟 150.0 → 149.3 ms，输出逐位不变；多分辨率进程 1326 → 1254 → 1162 MB；workspace 上限 2 GB → 64 MB 对 scratch 无影响（全是激活内存） |
| `final_1080p/` | 1080p 单分辨率最终配置 | 150.0 ms/帧（第二轮 152.0），显存峰值 1458 MB（1684），RSS 2470 MB |

回退 / 未采用：单一宽 profile（慢）、把画中画融合进整帧 kernel（负载不均，慢一倍多）、执行上下文在加载时全部创建（RSS +2 GB）、几何并行用"主线程等待时间"做判据（异步执行下误判）、batch=2（全部尺寸更慢）。
