"""Per-frame GPU timeline statistics from an nsys sqlite export of run_pipeline.py (NVTX frame<t>).
    nsys profile -t cuda,nvtx -o out python run_pipeline.py ... --max-frames 30
    nsys export -t sqlite -o out.sqlite out.nsys-rep && python3 prof/nsys_frames.py out.sqlite [skip]
Reports wall / GPU-busy per frame, copies, sync calls, and the largest GPU idle gaps with the
kernels around them (where the GPU waits for the CPU)."""
import sqlite3
import sys
from collections import Counter

db = sqlite3.connect(sys.argv[1])
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 5
q = lambda s: db.execute(s).fetchall()  # noqa: E731
names = dict(q("select id, value from StringIds"))
frames = sorted(db.execute("select start, end, text from NVTX_EVENTS where text like ? and end is not null", ("frame%",)).fetchall())
frames = frames[skip:]
kern = sorted((s, e, names.get(n, "?")) for s, e, n in q("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL"))
mem = q("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY")
rt = q("select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId = s.id")
ev = sorted([(s, e, "K:" + n) for s, e, n in kern] + [(s, e, "M:%d" % k) for s, e, _, k in mem])


def union(iv):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + ((cur_e - cur_s) if cur_e is not None else 0)


n = len(frames)
wall = sum(e - s for s, e, _ in frames) / n / 1e6
busy = sum(union([(max(s, fs), min(e, fe)) for s, e, _ in ev if s < fe and e > fs]) for fs, fe, _ in frames) / n / 1e6
print(f"frames {n}  wall/frame {wall:.1f} ms  GPU busy {busy:.1f} ms ({busy / wall * 100:.1f}%)")
kinds = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
for k, nm in kinds.items():
    sel = [(s, e, b) for s, e, b, kk in mem if kk == k and any(fs <= s < fe for fs, fe, _ in frames)]
    if sel:
        ms = sum(e - s for s, e, _ in sel) / n / 1e6
        mb = sum(b for *_, b in sel) / n / 2**20
        print(f"memcpy {nm}: {len(sel) / n:.0f}/frame {mb:.1f} MB/frame {ms:.2f} ms/frame")
for pat in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaLaunchKernel", "cudaMemcpyAsync"):
    sel = [(s, e) for s, e, nm in rt if nm.startswith(pat) and any(fs <= s < fe for fs, fe, _ in frames)]
    print(f"runtime {pat}: {len(sel) / n:.0f} calls/frame {sum(e - s for s, e in sel) / n / 1e6:.1f} ms/frame")
# idle gaps inside frames
gaps = Counter()
gap_ms = Counter()
for fs, fe, _ in frames:
    inside = [x for x in ev if x[0] < fe and x[1] > fs]
    end, prev = fs, "frame start"
    for s, e, nm in inside:
        if s - end > 200_000:  # > 0.2 ms
            key = f"{prev[:60]}  ->  {nm[:60]}"
            gaps[key] += 1
            gap_ms[key] += (s - end) / 1e6
        if e > end:
            end, prev = e, nm
    if fe - end > 200_000:
        gaps[f"{prev[:60]}  ->  frame end"] += 1
        gap_ms[f"{prev[:60]}  ->  frame end"] += (fe - end) / 1e6
print("largest GPU idle gaps (> 0.2 ms), ms/frame:")
for k, v in sorted(gap_ms.items(), key=lambda x: -x[1])[:15]:
    print(f"  {v / n:6.2f}  x{gaps[k] / n:.1f}  {k}")
