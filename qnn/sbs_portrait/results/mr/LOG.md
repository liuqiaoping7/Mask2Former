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
| `final_1080p/` | 1080p 单分辨率最终配置 | 150.0 ms/帧（第二轮 152.0），显存峰值 1458 MB（1684），RSS 2470 MB |

回退 / 未采用：单一宽 profile（慢）、把画中画融合进整帧 kernel（负载不均，慢一倍多）、执行上下文在加载时全部创建（RSS +2 GB）、几何并行用"主线程等待时间"做判据（异步执行下误判）、batch=2（全部尺寸更慢）。
