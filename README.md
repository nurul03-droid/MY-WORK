# GeoAI / CV Module — Preprocessing, Hough Detection, Vectorization

This module is owned by the **GeoAI/CV person** on the team. It takes a scanned
land blueprint image and outputs detected plot polygons, ready for the backend
person to plug into the FastAPI + PostGIS layer.

## Folder structure

```
geoai_cv_module/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Stage 1: deskew, denoise, binarize
│   ├── line_detection.py    # Stage 2: Hough line detection + gap closing
│   ├── vectorization.py     # Stage 3: contour extraction -> clean polygons
│   └── pipeline.py          # Wires stages 1-3 together, single entry point
├── tests/
│   └── test_pipeline.py     # Sanity tests using a synthetic test image
├── sample_data/
│   └── (put your test blueprint images here)
└── run_demo.py               # CLI script to run the pipeline on one image
```

## Setup

```bash
conda create -n geoai-cv python=3.11
conda activate geoai-cv
pip install -r requirements.txt
```

## Your responsibility (what to build/improve)

1. **`preprocessing.py`** — make sure blueprint scans of varying quality (skewed,
   low contrast, noisy) come out clean and binarized. This is the stage most
   worth tuning if detection quality is poor on real sample scans.
2. **`line_detection.py`** — tune Hough transform parameters
   (`threshold`, `minLineLength`, `maxLineGap`) and morphological closing
   kernel size per your actual sample blueprints. Different scan qualities
   need different values — there's no universal setting.
3. **`vectorization.py`** — controls how jagged pixel contours become clean
   polygons (`approxPolyDP` epsilon) and filters out noise-sized contours.

## Output contract (what you hand to the backend person)

`pipeline.detect_plots(image_path)` returns a list of dicts:

```python
[
  {
    "polygon_points": [[x1, y1], [x2, y2], ...],  # closed ring, pixel coords
    "area_px": 15234.5,                             # area in pixel^2
  },
  ...
]
```

The backend person converts `polygon_points` to real coordinates (via a
pixel-to-meter scale or GDAL georeferencing) and `area_px` isn't final area —
they'll recompute area after coordinate conversion. Don't do unit conversion
in this module; keep it purely pixel-space so it stays testable in isolation.

## Testing standalone (before handing off)

```bash
python run_demo.py sample_data/test_blueprint.png
```

This prints the detected polygons and area, and saves an overlay image
(`sample_data/test_blueprint_overlay.png`) so you can visually check detection
quality without needing the backend or database running at all.

## Tuning checklist if detection looks bad on real samples

- Lines missing/broken → increase `maxLineGap` in `line_detection.py`
- Too much noise detected → increase `min_area` threshold in `vectorization.py`
- Polygons too jagged → increase `epsilon` multiplier in `vectorization.py`
- Whole plots missed → lower `HoughLinesP` `threshold` (more sensitive, but noisier)
- **Duplicate/nested detections** (e.g. 2 real rectangles show up as 4) → this
  happens because a boundary line with thickness has an inner and outer edge,
  both detected as separate contours by `RETR_TREE`. The solution is to deduplicate
  nested polygons. Note that a threshold of ~15% area difference is recommended 
  instead of 5%, because thick boundary lines add significant area. Additionally,
  any contour that contains multiple valid plot holes (like the perimeter of adjacent 
  plots) must be completely filtered out.
- **Plots with broken/gapped boundaries** → increase `max_line_gap` in `line_detection.py` 
  (e.g., from 10 to 100) to allow the Hough transform to connect fragmented line 
  segments and properly close the plot polygon.
