"""
SQLite Database Layer for Agricultural Rover Ground Station
===========================================================
Starts completely empty (zero runs, zero captures, zero statistics) so all
displayed data, captures, heatmaps, and logs come strictly from real Gazebo
or physical rover trials.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")
DB_PATH = os.path.join(DATA_DIR, "ground_station.db")

DATASET_IMAGES_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR, "..", "..", "Final_Fruit_Dataset_YOLOv11", "test", "images"
    )
)

# Greenhouse CAD / Gazebo geometry matching agriculture_sim
FIELD_GEOMETRY = {
    "name": "Greenhouse Block A (2.0° Slope · 0.25° Uneven Aisles)",
    "x_min": -1.20,
    "x_max": 4.00,
    "y_min": -2.20,
    "y_max": 35.00,
    "row_y_min": -1.3925,
    "row_y_max": 33.6075,
    "rows": [
        {"index": 1, "x": 0.1018, "slope_deg": 0.25, "label": "Row 1"},
        {"index": 2, "x": 0.8585, "slope_deg": 0.25, "label": "Row 2"},
        {"index": 3, "x": 1.9418, "slope_deg": -0.25, "label": "Row 3"},
        {"index": 4, "x": 2.6985, "slope_deg": -0.25, "label": "Row 4"},
    ],
    "lanes": [
        {"name": "left", "x": -0.4400, "label": "Left Lane"},
        {"name": "center", "x": 1.4000, "label": "Center Lane"},
        {"name": "right", "x": 3.2400, "label": "Right Lane"},
    ],
    "aruco_y": [-1.0, 4.0, 9.0, 14.0, 19.0, 24.0, 29.0, 34.0],
    "mast_min_m": 0.7259,
    "mast_max_m": 1.1050,
}

DISEASE_META = {
    "anthracnose_fruit_rot": {
        "label": "Anthracnose Fruit Rot",
        "short": "Anthracnose",
        "prefix": "cls0_",
        "severity": "high",
        "color": "#dc2626",
    },
    "gray_mold": {
        "label": "Gray Mold (Botrytis)",
        "short": "Gray Mold",
        "prefix": "cls1_",
        "severity": "high",
        "color": "#d97706",
    },
    "powdery_mildew": {
        "label": "Powdery Mildew",
        "short": "Powdery Mildew",
        "prefix": "cls2_",
        "severity": "medium",
        "color": "#7c3aed",
    },
    "healthy": {
        "label": "Healthy Fruit",
        "short": "Healthy",
        "prefix": "cls3_",
        "severity": "none",
        "color": "#059669",
    },
}

YOLO_CLASS_ID_MAP = {
    0: "anthracnose_fruit_rot",
    1: "gray_mold",
    2: "powdery_mildew",
    3: "healthy",
}


def map_yolo_class(class_id: int, class_name: str = "") -> str:
    if class_id in YOLO_CLASS_ID_MAP:
        return YOLO_CLASS_ID_MAP[class_id]
    low = class_name.lower()
    if "anthracnose" in low:
        return "anthracnose_fruit_rot"
    if "gray" in low or "botrytis" in low or "mold" in low:
        return "gray_mold"
    if "powdery" in low or "mildew" in low:
        return "powdery_mildew"
    return "healthy"


def get_aruco_landmarks() -> list[dict[str, Any]]:
    markers = []
    marker_id = 0
    for lane in FIELD_GEOMETRY["lanes"]:
        for y in FIELD_GEOMETRY["aruco_y"]:
            markers.append(
                {
                    "id": marker_id,
                    "lane": lane["name"],
                    "x": lane["x"],
                    "y": y,
                }
            )
            marker_id += 1
    return markers


@contextmanager
def get_conn():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Initialize SQLite schema with ZERO hardcoded or seeded data."""
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                field_name TEXT NOT NULL,
                slope_deg REAL NOT NULL DEFAULT 2.0,
                distance_m REAL NOT NULL DEFAULT 0.0,
                duration_s REAL NOT NULL DEFAULT 0.0,
                total_detections INTEGER NOT NULL DEFAULT 0,
                diseased_count INTEGER NOT NULL DEFAULT 0,
                healthy_count INTEGER NOT NULL DEFAULT 0,
                avg_confidence REAL NOT NULL DEFAULT 0.0,
                avg_mast_height_m REAL NOT NULL DEFAULT 0.0,
                notes TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                timestamp TEXT NOT NULL,
                disease_class TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'medium',
                confidence REAL NOT NULL,
                x_m REAL NOT NULL,
                y_m REAL NOT NULL,
                row_index INTEGER NOT NULL,
                lane TEXT NOT NULL,
                camera_side TEXT NOT NULL,
                mast_height_m REAL NOT NULL,
                image_path TEXT NOT NULL,
                bbox_json TEXT DEFAULT '[]',
                reviewed INTEGER NOT NULL DEFAULT 0,
                operator_note TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS telemetry_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                timestamp TEXT NOT NULL,
                x_m REAL NOT NULL,
                y_m REAL NOT NULL,
                yaw_rad REAL NOT NULL,
                linear_vel REAL NOT NULL DEFAULT 0.0,
                angular_vel REAL NOT NULL DEFAULT 0.0,
                mast_height_m REAL NOT NULL DEFAULT 0.0,
                raw_odom_x REAL NOT NULL,
                raw_odom_y REAL NOT NULL,
                aruco_id INTEGER,
                battery_pct REAL NOT NULL DEFAULT 100.0
            );

            CREATE TABLE IF NOT EXISTS mission_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT 'info',
                category TEXT NOT NULL DEFAULT 'mission',
                message TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_detections_run ON detections(run_id);
            CREATE INDEX IF NOT EXISTS idx_telemetry_run ON telemetry_logs(run_id);
            CREATE INDEX IF NOT EXISTS idx_events_run ON mission_events(run_id);
            """
        )


def reset_database():
    """Erase all runs, detections, telemetry, events, and saved snapshots back to zero."""
    with get_conn() as conn:
        conn.executescript(
            """
            DELETE FROM detections;
            DELETE FROM telemetry_logs;
            DELETE FROM mission_events;
            DELETE FROM runs;
            DELETE FROM sqlite_sequence;
            """
        )
    if os.path.isdir(SNAPSHOTS_DIR):
        for fname in os.listdir(SNAPSHOTS_DIR):
            fpath = os.path.join(SNAPSHOTS_DIR, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass


def list_runs() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def get_run(run_id: int) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def get_active_run() -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE status = 'active' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def ensure_active_run(default_prefix: str = "Gazebo Live Trial") -> dict[str, Any]:
    active = get_active_run()
    if active is not None:
        return active
    count = len(list_runs()) + 1
    return create_run(
        f"{default_prefix} #{count:02d}",
        "Auto-started when live ROS 2 / Gazebo session connected.",
    )


def create_run(name: str, notes: str = "") -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE runs SET status = 'completed', ended_at = ? WHERE status = 'active'",
            (now_iso,),
        )
        cur = conn.execute(
            """
            INSERT INTO runs (name, started_at, status, field_name, slope_deg, notes)
            VALUES (?, ?, 'active', ?, 2.0, ?)
            """,
            (name, now_iso, FIELD_GEOMETRY["name"], notes),
        )
        run_id = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO mission_events (run_id, timestamp, level, category, message) VALUES (?, ?, 'info', 'mission', ?)",
            (run_id, now_iso, f"Started live trial run '{name}'."),
        )
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row)


def complete_run(run_id: int, status: str = "completed") -> dict[str, Any] | None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            return None
        try:
            started = datetime.fromisoformat(run["started_at"])
            dur = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
        except Exception:
            dur = run["duration_s"]
        conn.execute(
            "UPDATE runs SET status = ?, ended_at = ?, duration_s = ? WHERE id = ?",
            (status, now_iso, round(dur, 1), run_id),
        )
        conn.execute(
            "INSERT INTO mission_events (run_id, timestamp, level, category, message) VALUES (?, ?, 'info', 'mission', ?)",
            (run_id, now_iso, f"Run marked as {status}."),
        )
        updated = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(updated)


def delete_run(run_id: int) -> bool:
    with get_conn() as conn:
        # Remove associated snapshot files on disk
        snaps = conn.execute(
            "SELECT image_path FROM detections WHERE run_id = ?", (run_id,)
        ).fetchall()
        cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        deleted = cur.rowcount > 0
    if deleted:
        for s in snaps:
            rel = s["image_path"] or ""
            if rel.startswith("snapshots/"):
                fpath = os.path.join(SNAPSHOTS_DIR, os.path.basename(rel))
                if os.path.isfile(fpath):
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass
    return deleted


def add_event(run_id: int, level: str, category: str, message: str):
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO mission_events (run_id, timestamp, level, category, message) VALUES (?, ?, ?, ?, ?)",
            (run_id, now_iso, level, category, message),
        )


def add_detection(
    run_id: int,
    disease_class: str,
    confidence: float,
    x_m: float,
    y_m: float,
    row_index: int,
    lane: str,
    camera_side: str,
    mast_height_m: float,
    image_path: str,
    bbox: list | None = None,
) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    severity = DISEASE_META.get(disease_class, {}).get("severity", "medium")
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO detections (
                run_id, timestamp, disease_class, severity, confidence,
                x_m, y_m, row_index, lane, camera_side, mast_height_m,
                image_path, bbox_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                now_iso,
                disease_class,
                severity,
                round(confidence, 3),
                round(x_m, 4),
                round(y_m, 4),
                row_index,
                lane,
                camera_side,
                round(mast_height_m, 4),
                image_path,
                json.dumps(bbox or []),
            ),
        )
        det_id = int(cur.lastrowid)

        agg = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN disease_class != 'healthy' THEN 1 ELSE 0 END) AS diseased,
                SUM(CASE WHEN disease_class = 'healthy' THEN 1 ELSE 0 END) AS healthy,
                AVG(confidence) AS avg_conf,
                AVG(mast_height_m) AS avg_mast
            FROM detections WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        conn.execute(
            """
            UPDATE runs
            SET total_detections = ?, diseased_count = ?, healthy_count = ?,
                avg_confidence = ?, avg_mast_height_m = ?
            WHERE id = ?
            """,
            (
                agg["total"] or 0,
                agg["diseased"] or 0,
                agg["healthy"] or 0,
                round(agg["avg_conf"] or 0.0, 3),
                round(agg["avg_mast"] or 0.0, 3),
                run_id,
            ),
        )
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
        return dict(row)


def replace_detection_capture(
    det_id: int,
    run_id: int,
    disease_class: str,
    confidence: float,
    x_m: float,
    y_m: float,
    mast_height_m: float,
    image_path: str,
    bbox: list | None = None,
) -> dict[str, Any] | None:
    """Upgrade an existing panel capture (e.g. Healthy -> Diseased or higher confidence)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    severity = DISEASE_META.get(disease_class, {}).get("severity", "medium")
    old_img = None
    with get_conn() as conn:
        prev = conn.execute(
            "SELECT image_path FROM detections WHERE id = ? AND run_id = ?",
            (det_id, run_id),
        ).fetchone()
        if not prev:
            return None
        old_img = prev["image_path"]
        conn.execute(
            """
            UPDATE detections
            SET timestamp = ?, disease_class = ?, severity = ?, confidence = ?,
                x_m = ?, y_m = ?, mast_height_m = ?, image_path = ?, bbox_json = ?
            WHERE id = ?
            """,
            (
                now_iso,
                disease_class,
                severity,
                round(confidence, 3),
                round(x_m, 4),
                round(y_m, 4),
                round(mast_height_m, 4),
                image_path,
                json.dumps(bbox or []),
                det_id,
            ),
        )
        agg = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN disease_class != 'healthy' THEN 1 ELSE 0 END) AS diseased,
                SUM(CASE WHEN disease_class = 'healthy' THEN 1 ELSE 0 END) AS healthy,
                AVG(confidence) AS avg_conf,
                AVG(mast_height_m) AS avg_mast
            FROM detections WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        conn.execute(
            """
            UPDATE runs
            SET total_detections = ?, diseased_count = ?, healthy_count = ?,
                avg_confidence = ?, avg_mast_height_m = ?
            WHERE id = ?
            """,
            (
                agg["total"] or 0,
                agg["diseased"] or 0,
                agg["healthy"] or 0,
                round(agg["avg_conf"] or 0.0, 3),
                round(agg["avg_mast"] or 0.0, 3),
                run_id,
            ),
        )
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
        res = dict(row)

    if old_img and old_img != image_path and old_img.startswith("snapshots/"):
        fpath = os.path.join(SNAPSHOTS_DIR, os.path.basename(old_img))
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
            except OSError:
                pass
    return res


def add_telemetry(
    run_id: int,
    x_m: float,
    y_m: float,
    yaw_rad: float,
    linear_vel: float,
    angular_vel: float,
    mast_height_m: float,
    raw_odom_x: float,
    raw_odom_y: float,
    aruco_id: int | None = None,
    battery_pct: float = 100.0,
):
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    with get_conn() as conn:
        run = conn.execute("SELECT started_at FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            return
        last = conn.execute(
            "SELECT x_m, y_m FROM telemetry_logs WHERE run_id = ? ORDER BY id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        step_d = math.hypot(x_m - last["x_m"], y_m - last["y_m"]) if last else 0.0
        conn.execute(
            """
            INSERT INTO telemetry_logs (
                run_id, timestamp, x_m, y_m, yaw_rad, linear_vel, angular_vel,
                mast_height_m, raw_odom_x, raw_odom_y, aruco_id, battery_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                now_iso,
                round(x_m, 4),
                round(y_m, 4),
                round(yaw_rad, 4),
                round(linear_vel, 3),
                round(angular_vel, 3),
                round(mast_height_m, 4),
                round(raw_odom_x, 4),
                round(raw_odom_y, 4),
                aruco_id,
                round(battery_pct, 1),
            ),
        )
        try:
            dur = max(0.0, (now_dt - datetime.fromisoformat(run["started_at"])).total_seconds())
        except Exception:
            dur = 0.0
        inc_d = step_d if (0.005 < step_d < 3.0) else 0.0
        conn.execute(
            "UPDATE runs SET distance_m = ROUND(distance_m + ?, 2), duration_s = ? WHERE id = ?",
            (inc_d, round(dur, 1), run_id),
        )


def get_detections(
    run_id: int,
    disease_class: str | None = None,
    row_index: int | None = None,
    only_diseased: bool = False,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM detections WHERE run_id = ?"
    params: list[Any] = [run_id]
    if only_diseased:
        query += " AND disease_class != 'healthy'"
    if disease_class and disease_class != "all":
        query += " AND disease_class = ?"
        params.append(disease_class)
    if row_index is not None:
        query += " AND row_index = ?"
        params.append(row_index)
    query += " ORDER BY id DESC"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["bbox"] = json.loads(d.get("bbox_json") or "[]")
            except Exception:
                d["bbox"] = []
            d["reviewed"] = bool(d["reviewed"])
            result.append(d)
        return result


def update_detection(
    det_id: int, reviewed: bool | None = None, operator_note: str | None = None
) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
        if not row:
            return None
        new_rev = int(reviewed) if reviewed is not None else row["reviewed"]
        new_note = operator_note if operator_note is not None else row["operator_note"]
        conn.execute(
            "UPDATE detections SET reviewed = ?, operator_note = ? WHERE id = ?",
            (new_rev, new_note, det_id),
        )
        updated = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
        d = dict(updated)
        d["reviewed"] = bool(d["reviewed"])
        return d


def get_run_stats(run_id: int) -> dict[str, Any]:
    with get_conn() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            return {}
        dets = conn.execute(
            "SELECT * FROM detections WHERE run_id = ?", (run_id,)
        ).fetchall()
        telem = conn.execute(
            "SELECT * FROM telemetry_logs WHERE run_id = ? ORDER BY id ASC", (run_id,)
        ).fetchall()

        class_counts = {k: 0 for k in DISEASE_META}
        row_stats = {
            r["index"]: {
                "row": f"Row {r['index']}",
                "row_index": r["index"],
                "healthy": 0,
                "anthracnose_fruit_rot": 0,
                "gray_mold": 0,
                "powdery_mildew": 0,
                "diseased_total": 0,
                "total": 0,
            }
            for r in FIELD_GEOMETRY["rows"]
        }

        zone_bins = [
            {"zone": f"{i*5}–{(i+1)*5}m", "y_start": i * 5, "diseased": 0, "healthy": 0}
            for i in range(7)
        ]

        reviewed_diseased = 0
        for d in dets:
            cls = d["disease_class"]
            if cls in class_counts:
                class_counts[cls] += 1
            ri = d["row_index"]
            if ri in row_stats:
                row_stats[ri]["total"] += 1
                if cls in row_stats[ri]:
                    row_stats[ri][cls] += 1
                if cls != "healthy":
                    row_stats[ri]["diseased_total"] += 1
            if cls != "healthy" and d["reviewed"]:
                reviewed_diseased += 1

            z_idx = max(0, min(6, int(max(0.0, d["y_m"]) // 5.0)))
            if cls == "healthy":
                zone_bins[z_idx]["healthy"] += 1
            else:
                zone_bins[z_idx]["diseased"] += 1

        total = len(dets)
        diseased = sum(v for k, v in class_counts.items() if k != "healthy")
        healthy = class_counts["healthy"]
        health_score = round(100.0 * healthy / total, 1) if total > 0 else None

        drifts = [
            math.hypot(t["raw_odom_x"] - t["x_m"], t["raw_odom_y"] - t["y_m"])
            for t in telem
        ]
        max_drift_m = round(max(drifts), 3) if drifts else 0.0
        avg_drift_m = round(sum(drifts) / len(drifts), 3) if drifts else 0.0

        disease_breakdown = [
            {
                "key": k,
                "name": DISEASE_META[k]["short"],
                "full_name": DISEASE_META[k]["label"],
                "count": class_counts[k],
                "color": DISEASE_META[k]["color"],
                "pct": round(100.0 * class_counts[k] / total, 1) if total > 0 else 0.0,
            }
            for k in DISEASE_META
        ]

        return {
            "run": dict(run),
            "health_score_pct": health_score,
            "total_detections": total,
            "diseased_count": diseased,
            "healthy_count": healthy,
            "reviewed_diseased": reviewed_diseased,
            "disease_breakdown": disease_breakdown,
            "row_breakdown": list(row_stats.values()),
            "zone_profile": zone_bins,
            "odom_drift": {
                "max_drift_m": max_drift_m,
                "avg_drift_m": avg_drift_m,
                "samples": len(telem),
            },
        }


def get_empty_heatmap() -> dict[str, Any]:
    return {
        "field": FIELD_GEOMETRY,
        "aruco_landmarks": get_aruco_landmarks(),
        "detections": [],
        "ekf_path": [],
        "raw_odom_path": [],
    }


def get_run_heatmap(run_id: int) -> dict[str, Any]:
    with get_conn() as conn:
        dets = get_detections(run_id)
        telem = conn.execute(
            "SELECT x_m, y_m, yaw_rad, raw_odom_x, raw_odom_y, mast_height_m, aruco_id "
            "FROM telemetry_logs WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()

        ekf_path = [{"x": t["x_m"], "y": t["y_m"]} for t in telem]
        raw_path = [{"x": t["raw_odom_x"], "y": t["raw_odom_y"]} for t in telem]

        return {
            "field": FIELD_GEOMETRY,
            "aruco_landmarks": get_aruco_landmarks(),
            "detections": dets,
            "ekf_path": ekf_path,
            "raw_odom_path": raw_path,
        }


def get_run_events(run_id: int, category: str | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM mission_events WHERE run_id = ?"
    params: list[Any] = [run_id]
    if category and category != "all":
        query += " AND category = ?"
        params.append(category)
    query += " ORDER BY id DESC LIMIT 300"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def export_run_csv(run_id: int) -> str:
    dets = get_detections(run_id)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "id",
            "timestamp",
            "disease_class",
            "severity",
            "confidence",
            "x_m",
            "y_m",
            "row_index",
            "lane",
            "camera_side",
            "mast_height_m",
            "reviewed",
            "operator_note",
            "image_path",
        ]
    )
    for d in dets:
        writer.writerow(
            [
                d["id"],
                d["timestamp"],
                d["disease_class"],
                d["severity"],
                d["confidence"],
                d["x_m"],
                d["y_m"],
                d["row_index"],
                d["lane"],
                d["camera_side"],
                d["mast_height_m"],
                d["reviewed"],
                d["operator_note"],
                d["image_path"],
            ]
        )
    return buf.getvalue()
