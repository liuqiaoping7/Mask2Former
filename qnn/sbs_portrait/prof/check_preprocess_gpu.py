"""preprocess_gpu (OpenCV INTER_AREA weights on the GPU) vs preprocess (cv2.resize INTER_AREA)."""
import glob, os, sys, time
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import split_sbs
from seg import PersonSegmenter
files = sorted(glob.glob(os.path.expanduser("~/data/sbs/portrait_0121/*.jpg")))[::15]
seg = PersonSegmenter(precision="fp16", backbone="trt", eye_hw=(1080, 1920), pixel_decoder="trt")
m = seg.model
diff = []
for f in files:
    fr = cv2.imread(f); ft = torch.from_numpy(fr).cuda()
    for e, eye in enumerate(split_sbs(fr)):
        ref = seg.preprocess(eye).numpy().astype(np.int16)
        g = seg.preprocess_gpu(ft[:, e * 1920:(e + 1) * 1920])
        g = (g * m.pixel_std + m.pixel_mean).round().cpu().numpy().astype(np.int16)
        diff.append(np.abs(g - ref))
d = np.stack(diff)
print("shape", d.shape[1:], "max |diff|", int(d.max()), "pixels != :", float((d > 0).mean()), "mean", float(d.mean()))
ft = torch.from_numpy(cv2.imread(files[0])).cuda(); eye = split_sbs(cv2.imread(files[0]))[0]
for _ in range(5): seg.preprocess_gpu(ft[:, :1920])
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(50): seg.preprocess_gpu(ft[:, :1920])
torch.cuda.synchronize(); print("gpu ms/eye", (time.perf_counter() - t) / 50 * 1000)
t = time.perf_counter()
for _ in range(20): seg.preprocess(eye)
print("cpu ms/eye", (time.perf_counter() - t) / 20 * 1000)
