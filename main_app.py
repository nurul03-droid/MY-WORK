"""
FastAPI backend for Indian land blueprint detection and validation.

Wires together: file upload -> blueprint_geoai_pipeline -> Indian unit
conversion -> validation -> PostGIS storage -> GeoJSON response for the map UI.

Install:
    pip install fastapi uvicorn python-multipart sqlalchemy geoalchemy2 psycopg2-binary
    (plus everything blueprint_geoai_pipeline.py needs: opencv-python, shapely, pytesseract)

Run:
    uvicorn main_app:app --reload
"""

import json
import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text

from blueprint_geoai_pipeline import run_blueprint_pipeline

# ---------------------------------------------------------------------------
# Indian land-area unit conversion
# ---------------------------------------------------------------------------
# Bigha, katha, guntha vary by state — these are common approximations.
# For a production system, let the user also select the state and use the
# state-specific conversion factor instead of a single default.

UNIT_TO_SQM = {
    "sqm": 1.0,
    "sqft": 0.092903,
    "acre": 4046.86,
    "hectare": 10000.0,
    "guntha": 101.17,
    "bigha_up": 2529.29,      # Uttar Pradesh (pucca bigha)
    "bigha_wb": 1333.33,      # West Bengal
    "bigha_rajasthan": 1600.0,
    "katha_wb": 66.89,        # West Bengal katha
    "katha_bihar": 1361.29,   # Bihar katha (varies by district)
}


def to_sqm(value: float, unit: str) -> float:
    if unit not in UNIT_TO_SQM:
        raise HTTPException(400, f"Unknown unit '{unit}'. Supported: {list(UNIT_TO_SQM)}")
    return value * UNIT_TO_SQM[unit]


# ---------------------------------------------------------------------------
# App + DB setup
# ---------------------------------------------------------------------------

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Indian Land Blueprint Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Switch to SQLite for a seamless local demo
DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./landrecords.db")
# connect_args is needed for SQLite to allow multiple threads
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {})

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

with engine.begin() as conn:
    # Use standard SQLite types (INTEGER PRIMARY KEY) and avoid PostGIS specific types for the demo
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS parcels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            khasra_no TEXT,
            village TEXT,
            geom TEXT NULL,
            local_geom_json TEXT,
            area_measured_sqm FLOAT,
            area_declared_sqm FLOAT,
            declared_unit TEXT,
            flagged BOOLEAN DEFAULT FALSE,
            flag_reason TEXT
        );
    """))


# ---------------------------------------------------------------------------
# Endpoint: upload blueprint + land details -> detect -> validate -> store
# ---------------------------------------------------------------------------

@app.post("/detect-parcel")
async def detect_parcel(
    file: UploadFile = File(...),
    khasra_no: str = Form(...),
    village: str = Form(...),
    declared_area: float = Form(...),
    declared_unit: str = Form("sqm"),
    pixel_to_meter_ratio: float = Form(1.0),
):
    # 1. Save the uploaded blueprint
    ext = file.filename.split(".")[-1]
    save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.{ext}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2. Run the GeoAI detection pipeline
    results = run_blueprint_pipeline(save_path, pixel_to_meter_ratio=pixel_to_meter_ratio)
    if not results:
        raise HTTPException(422, "No plot boundary detected in the uploaded blueprint.")

    # For a demo, take the largest detected polygon as the target plot.
    # In a full system you'd let the user click/select which polygon matches their khasra number.
    best = max(results, key=lambda r: r["measured_area_sqm"])

    # 3. Convert declared area to sqm for comparison
    declared_sqm = to_sqm(declared_area, declared_unit)
    diff_ratio = abs(best["measured_area_sqm"] - declared_sqm) / declared_sqm if declared_sqm else 0
    flagged = diff_ratio > 0.10
    flag_reason = f"Area mismatch: {diff_ratio:.1%} difference from declared record" if flagged else None

    # 4. Store — geom is left NULL here since local pixel/metric coords aren't
    #    real lat/long; only insert into the `geom` PostGIS column once the
    #    blueprint has been tied to real coordinates via GCPs (see the main
    #    georeferencing step). Local coordinates are still stored as JSON so
    #    the UI can render the shape immediately.
    local_geom = json.dumps(list(best["polygon"].exterior.coords))

    with engine.begin() as conn:
        row = conn.execute(
            text("""
                INSERT INTO parcels (khasra_no, village, local_geom_json,
                                      area_measured_sqm, area_declared_sqm, declared_unit,
                                      flagged, flag_reason)
                VALUES (:khasra_no, :village, :local_geom, :measured, :declared, :unit, :flagged, :reason)
                RETURNING id
            """),
            {
                "khasra_no": khasra_no,
                "village": village,
                "local_geom": local_geom,
                "measured": best["measured_area_sqm"],
                "declared": declared_sqm,
                "unit": declared_unit,
                "flagged": flagged,
                "reason": flag_reason,
            },
        ).fetchone()

    return {
        "id": row[0],
        "khasra_no": khasra_no,
        "village": village,
        "local_polygon": list(best["polygon"].exterior.coords),
        "area_measured_sqm": round(best["measured_area_sqm"], 2),
        "area_declared_sqm": round(declared_sqm, 2),
        "flagged": flagged,
        "flag_reason": flag_reason,
    }


@app.get("/parcels")
def list_parcels():
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, khasra_no, village, local_geom_json,
                   area_measured_sqm, area_declared_sqm, flagged, flag_reason
            FROM parcels ORDER BY id DESC
        """)).mappings().all()
    return [dict(r) | {"local_geom_json": json.loads(r["local_geom_json"])} for r in rows]

# Mount the frontend directory to serve the UI
import os
os.makedirs("frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
