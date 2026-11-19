"""Analyze the nsys sqlite export of prof_pipeline.py (PROF_NOSYNC=1, NVTX ranges).

usage: python nsys_analyze.py pipe.sqlite
Reports per-frame GPU busy ratio, per-NVTX-range GPU busy / kernel gaps, memcpy
volume and bandwidth, blocking CUDA runtime calls and top CPU sample symbols.
"""
import collections
import sqlite3
import sys

c = sqlite3.connect(sys.argv[1])
tabs = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
strs = dict(c.execute("select id, value from StringIds"))
nv = [(s, e, t or strs.get(tid)) for s, e, t, tid in
      c.execute("select start, end, text, textId from NVTX_EVENTS where end is not null")]
K = sorted(c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL"))
MC = sorted(c.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY")) if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs else []
G = sorted(K + MC)


def busy(s, e):
    tot, cur = 0, s
    for a, b in G:
        if a >= e:
            break
        a2, b2 = max(a, cur), min(b, e)
        if b2 > a2:
            tot += b2 - a2
            cur = b2
    return tot


def gaps(s, e):
    ks = [(a, b) for a, b in K if a >= s and b <= e]
    g = [ks[i + 1][0] - ks[i][1] for i in range(len(ks) - 1)]
    return len(ks), sum(x for x in g if x > 0), sum(1 for x in g if x > 20000)


frames = sorted([(s, e) for s, e, t in nv if t and t.startswith("frame")])[3:]  # drop warm-up
nf = len(frames)
t0, t1 = frames[0][0], frames[-1][1]
fw = sum(e - s for s, e in frames) / nf
fb = sum(busy(s, e) for s, e in frames) / nf
print(f"frames={nf}  wall={fw / 1e6:.1f} ms  GPU busy={fb / 1e6:.1f} ms ({100 * fb / fw:.1f}%)")

agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
for s, e, t in nv:
    if not t or t.startswith("frame") or t.startswith("cub::") or s < t0 or e > t1:
        continue
    a = agg[t]
    n, gsum, big = gaps(s, e)
    a[0] += e - s; a[1] += busy(s, e); a[2] += n; a[3] += gsum; a[4] += big
print(f"{'range':40s} {'cpu wall ms':>11s} {'gpu ms':>8s} {'busy%':>6s} {'kernels':>8s} {'gap ms':>7s} {'>20us':>6s}")
for t, a in sorted(agg.items(), key=lambda x: -x[1][0]):
    print(f"{t:40s} {a[0] / nf / 1e6:11.1f} {a[1] / nf / 1e6:8.1f} {100 * a[1] / max(a[0], 1):5.1f}% "
          f"{a[2] / nf:8.0f} {a[3] / nf / 1e6:7.2f} {a[4] / nf:6.0f}")

names = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
by = collections.defaultdict(lambda: [0, 0, 0])
for k, b, s, e in c.execute("select copyKind, bytes, start, end from CUPTI_ACTIVITY_KIND_MEMCPY"):
    if s >= t0 and e <= t1:
        by[k][0] += 1; by[k][1] += b; by[k][2] += e - s
for k, v in by.items():
    print(f"memcpy {names.get(k, k)}: {v[0] / nf:.0f}/frame, {v[1] / nf / 1e6:.1f} MB/frame, "
          f"{v[2] / nf / 1e6:.2f} ms/frame, {v[1] / max(v[2], 1):.1f} GB/s")

rt = collections.defaultdict(lambda: [0, 0])
for nid, s, e in c.execute("select nameId, start, end from CUPTI_ACTIVITY_KIND_RUNTIME"):
    if s >= t0 and e <= t1:
        n = strs.get(nid, "?")
        if any(x in n for x in ("Synchronize", "Memcpy", "LaunchKernel")):
            key = "cudaLaunchKernel*" if "LaunchKernel" in n else n.split("_v")[0]
            rt[key][0] += 1; rt[key][1] += e - s
for k, v in sorted(rt.items(), key=lambda x: -x[1][1]):
    print(f"runtime {k:28s} {v[0] / nf:6.0f} calls/frame {v[1] / nf / 1e6:8.1f} ms/frame")

if "SAMPLING_CALLCHAINS" in tabs:
    q = ("select s.symbol, s.module from SAMPLING_CALLCHAINS s join COMPOSITE_EVENTS ce on s.id = ce.id "
         "where s.stackDepth = 0 and ce.start >= ? and ce.start <= ?")
    cnt = collections.Counter()
    for sym, mod in c.execute(q, (t0, t1)):
        cnt[strs.get(mod, "?").split("/")[-1].split(".so")[0][:28]] += 1
    tot = sum(cnt.values())
    print(f"CPU samples in window: {tot}  (by module)")
    for mod, n in cnt.most_common(12):
        print(f"  {100 * n / tot:5.1f}%  {mod}")
