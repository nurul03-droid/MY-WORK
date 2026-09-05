"""
Run this to test the pipeline on one image, standalone — no backend or
database needed. Saves an overlay image so you can visually check detection
quality.

Usage:
    python run_demo.py sample_data/test_blueprint.png
"""

import argparse
import json
import os

import cv2
import numpy as np

from src.pipeline import detect_plots


def draw_overlay(image_path: str, plots: list[dict], out_path: str):
    img = cv2.imread(image_path)
    for i, plot in enumerate(plots):
        pts = np.array(plot["polygon_points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        centroid = pts.mean(axis=0).astype(int)[0]
        cv2.putText(img, f"#{i}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)
    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone CV pipeline test")
    parser.add_argument("image", help="Path to a blueprint image")
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--epsilon-ratio", type=float, default=0.01)
    args = parser.parse_args()

    plots = detect_plots(args.image, min_area=args.min_area, epsilon_ratio=args.epsilon_ratio)

    print(f"Detected {len(plots)} plot(s):")
    for i, p in enumerate(plots):
        print(json.dumps({"id": i, "area_px": round(p["area_px"], 2)}, indent=2))

    base, ext = os.path.splitext(args.image)
    overlay_path = f"{base}_overlay{ext}"
    draw_overlay(args.image, plots, overlay_path)
    print(f"\nOverlay saved to: {overlay_path}")
