"""dilate_ellipse vs conv2d(ellipse) > 0 (must be identical) and separable vs 2-D box filter."""
import os, sys, cv2, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_post import GPUPost
bg = np.zeros((1080, 1920, 3), np.uint8)
gp = GPUPost((960, 540), bg, bg)
ell = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)).astype(np.float32))[None, None].cuda()
g = torch.Generator(device="cuda").manual_seed(0)
bad = 0
for density in (0.0005, 0.01, 0.2, 0.9):
    m = torch.rand(540, 960, device="cuda", generator=g) < density
    ref = F.conv2d(m.float()[None, None], ell, padding=15)[0, 0] > 0
    bad += int((gp.dilate_ellipse(m) != ref).sum())
m = torch.zeros(540, 960, dtype=torch.bool, device="cuda"); m[100:400, 300:600] = True  # blob + borders
m[0, 0] = m[-1, -1] = True
bad += int((gp.dilate_ellipse(m) != (F.conv2d(m.float()[None, None], ell, padding=15)[0, 0] > 0)).sum())
x = torch.rand(2, 540, 960, device="cuda", generator=g)
r = 4
ref = F.avg_pool2d(F.pad(x[:, None], (r, r, r, r), mode="reflect"), 2 * r + 1, stride=1)[:, 0]
print("dilation mismatches", bad, "| box max abs diff", float((gp.box(x) - ref).abs().max()))
def ev(fn, n=50):
    import time
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return round((time.perf_counter() - t) / n * 1000, 3)
print("dilate ms", ev(lambda: gp.dilate_ellipse(m)), "conv ms", ev(lambda: F.conv2d(m.float()[None, None], ell, padding=15)))
