"""Aggregate ncu raw CSV (one-eye forward, NVTX-scoped) into per-stage SOL / bound class."""
import csv
import re
import sys
from collections import defaultdict

rows = list(csv.reader(open(sys.argv[1])))
hdr_i = next(i for i, r in enumerate(rows) if r and r[0] == "ID")
hdr, data = rows[hdr_i], [r for r in rows[hdr_i + 2:] if r and r[0].isdigit()]
col = {n: i for i, n in enumerate(hdr)}
nv_col = next(i for i, n in enumerate(hdr) if n.startswith("thread Domain"))
M = {"dur": "gpu__time_duration.sum", "sm": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
     "mem": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
     "dram": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
     "tc": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"}
unit = rows[hdr_i + 1][col[M["dur"]]]
scale = {"nsecond": 1e-6, "usecond": 1e-3, "msecond": 1.0}.get(unit, 1e-6)


def f(r, k):
    v = r[col[M[k]]].replace(",", "")
    try:
        return float(v)
    except ValueError:
        return 0.0


def stage_of(r):
    found = re.findall(r"(S\d_[A-Za-z0-9_]+)", r[nv_col])
    return found[-1] if found else "?"


def klass(sm, mem):
    if sm >= 60 and sm >= mem:
        return "compute"
    if mem >= 60 and mem > sm:
        return "memory"
    return "latency"


def kname(n):
    n = re.sub(r"\(.*", "", n)
    return n.replace("void ", "").replace("at::native::", "")[:58]


agg = defaultdict(lambda: {"ms": 0.0, "n": 0, "sm": 0.0, "mem": 0.0, "dram": 0.0, "tc": 0.0,
                           "cls": defaultdict(float), "k": defaultdict(lambda: [0.0, 0, 0.0, 0.0])})
for r in data:
    s = stage_of(r)
    d = f(r, "dur") * scale
    a = agg[s]
    a["ms"] += d; a["n"] += 1
    for k in ("sm", "mem", "dram", "tc"):
        a[k] += f(r, k) * d
    a["cls"][klass(f(r, "sm"), f(r, "mem"))] += d
    kk = a["k"][kname(r[col["Kernel Name"]])]
    kk[0] += d; kk[1] += 1; kk[2] += f(r, "sm") * d; kk[3] += f(r, "mem") * d

tot = sum(a["ms"] for a in agg.values())
print(f"total kernel time (ncu, 1 eye, clock-control none): {tot:.1f} ms, kernels: {sum(a['n'] for a in agg.values())}")
print("| stage | kernels | ms | % | SM% | Mem% | DRAM% | TensorCore% | compute | memory | latency |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for s in sorted(agg):
    a = agg[s]; w = a["ms"] or 1
    c = a["cls"]
    print(f"| {s} | {a['n']} | {a['ms']:.1f} | {100 * a['ms'] / tot:.1f}% | {a['sm'] / w:.0f} | {a['mem'] / w:.0f} | "
          f"{a['dram'] / w:.0f} | {a['tc'] / w:.1f} | {100 * c['compute'] / w:.0f}% | {100 * c['memory'] / w:.0f}% | "
          f"{100 * c['latency'] / w:.0f}% |")
print()
for s in sorted(agg):
    a = agg[s]
    print(f"## {s} top kernels")
    for name, (ms, n, smw, memw) in sorted(a["k"].items(), key=lambda x: -x[1][0])[:6]:
        print(f"  {ms:7.2f} ms  x{n:<4d} SM {smw / ms:4.0f}%  Mem {memw / ms:4.0f}%  {klass(smw / ms, memw / ms):8s} {name}")
all_cls = defaultdict(float)
for a in agg.values():
    for k, v in a["cls"].items():
        all_cls[k] += v
print("\noverall:", {k: f"{100 * v / tot:.1f}%" for k, v in all_cls.items()})
