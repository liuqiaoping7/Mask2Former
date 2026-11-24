"""Markdown tables of the multi-resolution results (results/mr/*) for the docs.

  python prof/mr_tables.py            # all blocks, separated by '@@ NAME' lines
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "mr")
SIZES = ["640x480", "676x504", "854x480", "960x540", "1280x720", "1252x862", "1396x842", "1440x1080", "1680x966", "1920x1080"]
Q = ("tiou", "tflk_band", "siou", "sflk_band", "ghost")


def j(*p):
    f = os.path.join(R, *p)
    return json.load(open(f)) if os.path.exists(f) else None


def pnames(run):
    import trt_common as tc
    return list(tc.PROFILE_SETS[j(run, "summary.json")["profiles"]])


def q3d(ev, ref):
    q, qr = ev["quality"], ref["quality"]
    bad = [k for k in Q if abs(q[k] - qr[k]) > 0.05 * abs(qr[k])]
    return "±5% 内" if not bad else "、".join(f"{k} {qr[k]:.4f}→{q[k]:.4f}" for k in bad)


def final_rows(run="opt_final"):
    names = pnames(run)
    for s in SIZES:
        st, ev, rs, rq = j(run, s, "stats.json"), j(run, s, "eval.json"), j("ref_official", s, "stats.json"), j("ref_official", s, "eval.json")
        e = ev["error_vs_ref"]
        mi, pd = st["model_input_hw"], st["padded_hw"]
        print(f"| {s} | {mi[0]}x{mi[1]}（{pd[0]}x{pd[1]}） | {names[st['profile']]} | {st['e2e_ms']['median']:.1f} | {st['fps']:.2f} | "
              f"{rs['e2e_ms']['median']:.0f} | {e['alpha_mae_mean']:.1e} / {e['alpha_iou_mean']:.4f} / {e['alpha_iou_min']:.4f} | {q3d(ev, rq)} |")


def profile_rows():
    sets = [j(f"profile_spike_{k}.json") for k in ("static1080", "wide", "multires", "exact")]
    for s in ["768x1344", "800x1344", "800x1184", "800x1088", "768x1152", "736x1280", "544x960", "512x704", "480x864", "480x640"]:
        cells = []
        for d in sets:
            v = d.get(s) if d else None
            cells.append(f"{v['chain']:.1f}" if v else "–")
        print(f"| {s} | " + " | ".join(cells) + " |")


def aligned_rows():
    for s in ["640x480", "676x504", "1252x862", "1396x842", "1440x1080"]:
        o, a, ev, rq = j("opt_final", s, "stats.json"), j("opt_final_aligned", s, "stats.json"), j("opt_final_aligned", s, "eval.json"), j("ref_official", s, "eval.json")
        e = ev["error_vs_ref"]
        print(f"| {s} | {o['padded_hw'][0]}x{o['padded_hw'][1]} → {a['model_input_hw'][0]}x{a['model_input_hw'][1]} | {o['e2e_ms']['median']:.1f} | "
              f"{a['e2e_ms']['median']:.1f}（{a['e2e_ms']['median'] / o['e2e_ms']['median'] - 1:+.1%}） | {e['alpha_mae_mean']:.1e} | "
              f"{e['alpha_iou_mean']:.4f} / {e['alpha_iou_min']:.4f} | {q3d(ev, rq)} |")


def native_rows():
    for s in SIZES:
        o, n, ev, rq = j("opt_final", s, "stats.json"), j("opt_final_native", s, "stats.json"), j("opt_final_native", s, "eval.json"), j("ref_official", s, "eval.json")
        e = ev["error_vs_ref"]
        mi = n["model_input_hw"]
        print(f"| {s} | {mi[0]}x{mi[1]} | {o['e2e_ms']['median']:.1f} | {n['e2e_ms']['median']:.1f} | {o['e2e_ms']['median'] / n['e2e_ms']['median']:.2f}× | "
              f"{e['alpha_mae_mean']:.1e} | {e['alpha_iou_mean']:.4f} / {e['alpha_iou_min']:.4f} | {q3d(ev, rq)} |")


def geo_rows():
    for s in ["640x480", "676x504", "854x480", "960x540", "1280x720", "1920x1080"]:
        a, b, c = j("opt_native", s, "stats.json"), j("opt_native_geo", s, "stats.json"), j("opt_final_native", s, "stats.json")
        print(f"| {s} | {a['e2e_ms']['median']:.1f} | {b['e2e_ms']['median']:.1f} | {c['e2e_ms']['median']:.1f}（{'并行' if c.get('geo_parallel') else '串行'}） |")


def batch_rows():
    d = j("batch", "batch_spike.json")
    print("| 模型输入 | 两次 batch=1 ms（backbone / pixel decoder / decoder） | 一次 batch=2 ms | 差 | 激活显存 MB（b1 → b2，三段） | mask 正负号一致率 |")
    print("|---|---|---|---|---|---|")
    for s, v in d.items():
        a, b = v["b1_two_calls"], v["b2_one_call"]
        print(f"| {s} | {v['b1_chain']:.1f}（{a['backbone']:.1f} / {a['pixel_decoder']:.1f} / {a['decoder']:.2f}） | "
              f"{v['b2_chain']:.1f}（{b['backbone']:.1f} / {b['pixel_decoder']:.1f} / {b['decoder']:.2f}） | {v['b2_vs_b1']:+.1%} | "
              f"{sum(v['act_mem_mb']['b1']):.0f} → {sum(v['act_mem_mb']['b2']):.0f} | {v['mask_sign_agree']:.4f} |")


def pip_table(size="1920x1080"):
    d = j("pip", "pip_spike.json")[size]
    kinds = ["torch", "triton", "cuda_fused", "cuda", "cuda_tex", "cuda_tex1"]
    print("| 实现 | 仅合成 ms | 合成 + 本地 SBS 画中画 ms | 误差（最大 / 画中画 PSNR） | 合成 + 3D 图标 ms | 误差（最大 / 画中画 PSNR） |")
    print("|---|---|---|---|---|---|")
    f = lambda v: "–" if v is None else f"{v['kernel_ms']:.3f}"  # noqa: E731
    g = lambda v: "–" if v is None else f"{v['max_abs']} / {'∞' if v['pip_psnr_db'] == float('inf') else format(v['pip_psnr_db'], '.1f')} dB"  # noqa: E731
    for k in kinds:
        n, fr, ic = d.get(f"none/{k}"), d.get(f"frame/{k}"), d.get(f"icon/{k}")
        print(f"| {k} | {f(n)} | {f(fr)} | {g(fr)} | {f(ic)} | {g(ic)} |")


def pip_all_sizes():
    d = j("pip", "pip_spike.json")
    print("| 每眼分辨率 | 画中画 | torch | triton | cuda | cuda_tex（本地 SBS / 图标） |")
    print("|---|---|---|---|---|---|")
    for s in SIZES:
        v = d[s]
        print(f"| {s} | {v['pip_hw'][0]}x{v['pip_hw'][1]} | {v['frame/torch']['kernel_ms']:.3f} | {v['frame/triton']['kernel_ms']:.3f} | "
              f"{v['frame/cuda']['kernel_ms']:.3f} | {v['frame/cuda_tex']['kernel_ms']:.3f} / {v['icon/cuda_tex']['kernel_ms']:.3f} |")


def pip_e2e():
    cfgs = [("pip_torch_none", "torch 合成，无画中画"), ("pip_triton_none", "triton 合成，无画中画"), ("pip_triton_frame", "triton + 本地 SBS 画中画"),
            ("pip_triton_icon", "triton + 3D 图标"), ("pip_torch_frame", "torch + 本地 SBS 画中画"), ("pip_cuda_tex_icon", "cuda_tex + 3D 图标")]
    print("| 配置 | 1920x1080 ms | 1280x720 ms | 640x480 ms |")
    print("|---|---|---|---|")
    for n, label in cfgs:
        s = j("pip", "e2e", n, "summary.json")["sizes"]
        print(f"| {label} | " + " | ".join(f"{s[k]['e2e_ms']['median']:.1f}" for k in ("1920x1080", "1280x720", "640x480")) + " |")


if __name__ == "__main__":
    for name, fn in [("FINAL_ROWS", final_rows), ("PROFILE_ROWS", profile_rows), ("ALIGNED_ROWS", aligned_rows),
                     ("NATIVE_ROWS", native_rows), ("GEO_ROWS", geo_rows), ("BATCH_TABLE", batch_rows),
                     ("PIP_TABLE", pip_table), ("PIP_ALL", pip_all_sizes), ("PIP_E2E", pip_e2e)]:
        print(f"@@ {name}")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {e!r}")
