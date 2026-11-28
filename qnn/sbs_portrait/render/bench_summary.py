"""Medians over the rounds of render/bench.sh (markdown tables)."""
import json
import os
import sys
from glob import glob

import numpy as np

out = sys.argv[1]
med = lambda v: float(np.median(v))  # noqa: E731


def rows(prefix):
    return sorted(glob(os.path.join(out, prefix)))


print("| 流水线 | 配置 | ms/帧 | p95 | 发布→present返回 ms | 绘制 CPU ms | 绘制 GPU ms | 呈现/生产 | 显存 MB |")
print("|---|---|---|---|---|---|---|---|---|")
for lang in ("cpp", "python"):
    for cfg in ("none", "gl_zerocopy", "gl_copy", "vulkan_zerocopy", "vulkan_copy"):
        st = [json.load(open(os.path.join(d, "stats.json"))) for d in rows(f"pipe_{lang}_{cfg}_[0-9]")
              if os.path.isdir(d)]
        if not st:
            continue
        r = [s["render"] for s in st if "render" in s]
        get = lambda f: f"{med([f(x) for x in r]):.2f}" if r else "-"  # noqa: E731
        shown = f"{sum(x['presented'] for x in r)}/{sum(x['produced'] for x in r)}" if r else "-"
        print(f"| {lang} | {cfg} | {med([s['e2e_ms']['median'] for s in st]):.2f} "
              f"| {med([s['e2e_ms']['p95'] for s in st]):.1f} | {get(lambda x: x['latency_ms']['median'])} "
              f"| {get(lambda x: x['draw_cpu_ms'])} | {get(lambda x: x['draw_gpu_ms'])} | {shown} "
              f"| {med([s['gpu_mem_peak_mb_nvml'] for s in st]):.0f} |")
print()
print("| 仅渲染路径 | 帧 | 配置 | 生产者 ms/帧 | 发布→present返回 ms | 绘制 CPU ms | 绘制 GPU ms |")
print("|---|---|---|---|---|---|---|")
for lang in ("cpp", "python"):
    for size in ("1920x1080", "640x480"):
        for api in ("gl", "vulkan"):
            for present in ("zerocopy", "copy"):
                st = [json.load(open(f)) for f in rows(f"path_{lang}_{api}_{present}_{size}_*.json")]
                if not st:
                    continue
                g = lambda f: f"{med([f(s) for s in st]):.3f}"  # noqa: E731
                print(f"| {lang} | {st[0]['size']} | {api}_{present} | {g(lambda s: s['producer_ms_per_frame'])} "
                      f"| {g(lambda s: s['render']['latency_ms']['median'])} | {g(lambda s: s['render']['draw_cpu_ms'])} "
                      f"| {g(lambda s: s['render']['draw_gpu_ms'])} |")
