"""Rank SVD clips: video-conference-like portrait clips vs. person-free background clips.

Runs the person segmenter on the left view of each clip's first SBS frame.
Conference-like = 1-2 persons, the largest one covers 8-60% of the frame,
horizontally centred and reaching the upper half of the image (head visible).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from seg import DEFAULT_CONFIG, DEFAULT_WEIGHTS, PersonSegmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumbs", required=True, help="dir of <id>.jpg SBS first frames")
    ap.add_argument("--config-file", default=DEFAULT_CONFIG)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seg = PersonSegmenter(args.config_file, args.weights)

    rows = []
    for f in sorted(glob.glob(os.path.join(args.thumbs, "*.jpg"))):
        sbs = cv2.imread(f)
        w = sbs.shape[1] // 2
        left = sbs[:, :w]
        prob, hard = seg([left])[0]
        n, lab, stats, _ = cv2.connectedComponentsWithStats(hard.astype(np.uint8))
        comps = [s for s in stats[1:] if s[4] > 0.01 * hard.size]
        area = hard.mean()
        row = {"id": os.path.basename(f)[:-4], "person_area": float(area), "n_person_blobs": len(comps)}
        if comps:
            x, y, bw, bh, a = max(comps, key=lambda s: s[4])
            cx = (x + bw / 2) / hard.shape[1]
            row.update(cx=float(cx), top=float(y / hard.shape[0]), main_area=float(a / hard.size))
            conf_like = (1 <= len(comps) <= 2 and 0.08 <= a / hard.size <= 0.6
                         and 0.2 <= cx <= 0.8 and y / hard.shape[0] < 0.5)
            row["score"] = float(a / hard.size) if conf_like else 0.0
        else:
            row["score"] = 0.0
        rows.append(row)
        print(row, flush=True)

    portraits = sorted([r for r in rows if r["score"] > 0], key=lambda r: -r["score"])
    backgrounds = sorted([r for r in rows if r["person_area"] < 0.001], key=lambda r: r["id"])
    with open(os.path.join(args.out, "clip_ranking.json"), "w") as fp:
        json.dump({"portrait_candidates": portraits, "background_candidates": backgrounds,
                   "all": rows}, fp, indent=1)

    def sheet(items, name, cols=4):
        tiles = []
        for r in items[:16]:
            im = cv2.resize(cv2.imread(os.path.join(args.thumbs, r["id"] + ".jpg")), (640, 180))
            cv2.putText(im, r["id"], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            tiles.append(im)
        if not tiles:
            return
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(os.path.join(args.out, name), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])

    sheet(portraits, "portrait_candidates.jpg")
    sheet(backgrounds, "background_candidates.jpg")
    print(f"portraits={len(portraits)} backgrounds={len(backgrounds)}")


if __name__ == "__main__":
    main()
