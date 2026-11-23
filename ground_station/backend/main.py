"""
AgriRover Ground Station FastAPI Backend
========================================
Provides:
  - REST API for SQLite inspection runs, field statistics, disease detections,
    2D greenhouse heatmap data, event logs, database reset, and CSV export.
  - Real-time WebSocket (/ws/telemetry) for rover pose, EKF drift, mast height & status.
  - Live GStreamer 1.24 MJPEG endpoints (/api/stream/{left|right|bottom}) + custom
    GStreamer pipeline configuration for field deployment.
  - Live bidirectional ROS 2 robot controls (/api/control/*).
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database as db
from ros_bridge import RosBridgeManager

gst_manager = None
ros_bridge: RosBridgeManager | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global gst_manager, ros_bridge
    db.init_db()
    # Initialize ROS 2 bridge BEFORE importing/initializing GStreamer pipelines
    ros_bridge = RosBridgeManager()
    from gstreamer_streamer import GStreamerManager

    gst_manager = GStreamerManager()
    ros_bridge.attach_gst_manager(gst_manager)
    yield
    if ros_bridge:
        ros_bridge.shutdown()
    if gst_manager:
        gst_manager.shutdown()


app = FastAPI(title="AgriRover Ground Station API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Request Models ----------------
class CreateRunReq(BaseModel):
    name: str
    notes: str = ""


class CompleteRunReq(BaseModel):
    status: str = "completed"


class UpdateDetectionReq(BaseModel):
    reviewed: bool | None = None
    operator_note: str | None = None


class CmdVelReq(BaseModel):
    linear_x: float
    angular_z: float
    duration_s: float = 0.55


class EstopReq(BaseModel):
    active: bool


class MastHeightReq(BaseModel):
    height_m: float
    auto_align: bool | None = None


class MissionReq(BaseModel):
    action: str  # start | pause | resume | rtb | stop | reset
    run_id: int | None = None


class GstPipelineReq(BaseModel):
    channel: str  # left | right | bottom
    pipeline: str


class SnapshotReq(BaseModel):
    run_id: int | None = None
    camera: str = "right"
    disease_class: str = "gray_mold"
    confidence: float = 0.90


# ---------------- Run & Database Endpoints ----------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gst": gst_manager.get_all_status() if gst_manager else {},
        "telemetry": ros_bridge.get_telemetry() if ros_bridge else {},
        "field": db.FIELD_GEOMETRY,
    }


@app.get("/api/field/empty_heatmap")
def api_empty_heatmap():
    return db.get_empty_heatmap()


@app.post("/api/database/reset")
def api_reset_database():
    db.reset_database()
    if ros_bridge:
        ros_bridge.clear_active_run()
    return {"reset": True}


@app.get("/api/runs")
def api_list_runs():
    return {"runs": db.list_runs(), "field": db.FIELD_GEOMETRY}


@app.post("/api/runs")
def api_create_run(req: CreateRunReq):
    run = db.create_run(
        req.name.strip() or f"Trial Run {datetime.now().strftime('%H:%M')}", req.notes
    )
    if ros_bridge:
        ros_bridge.clear_active_run()
        ros_bridge.active_run_id = int(run["id"])
        ros_bridge.handle_mission_action("start", explicit_run_id=run["id"])
    return run


@app.patch("/api/runs/{run_id}/complete")
def api_complete_run(run_id: int, req: CompleteRunReq):
    run = db.complete_run(run_id, req.status)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if ros_bridge and ros_bridge.active_run_id == run_id:
        ros_bridge.handle_mission_action("stop")
        ros_bridge.clear_active_run()
    return run


@app.delete("/api/runs/{run_id}")
def api_delete_run(run_id: int):
    if not db.delete_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    if ros_bridge and ros_bridge.active_run_id == run_id:
        ros_bridge.clear_active_run()
    return {"deleted": True, "id": run_id}


@app.get("/api/runs/{run_id}/stats")
def api_run_stats(run_id: int):
    stats = db.get_run_stats(run_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Run not found")
    return stats


@app.get("/api/runs/{run_id}/heatmap")
def api_run_heatmap(run_id: int):
    if not db.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return db.get_run_heatmap(run_id)


@app.get("/api/runs/{run_id}/detections")
def api_run_detections(
    run_id: int,
    disease_class: str | None = Query(default=None),
    row_index: int | None = Query(default=None),
    only_diseased: bool = Query(default=False),
):
    if not db.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "detections": db.get_detections(
            run_id,
            disease_class=disease_class,
            row_index=row_index,
            only_diseased=only_diseased,
        ),
        "disease_meta": db.DISEASE_META,
    }


@app.patch("/api/detections/{det_id}")
def api_update_detection(det_id: int, req: UpdateDetectionReq):
    updated = db.update_detection(
        det_id, reviewed=req.reviewed, operator_note=req.operator_note
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Detection not found")
    return updated


@app.get("/api/runs/{run_id}/logs")
def api_run_logs(run_id: int, category: str | None = Query(default=None)):
    if not db.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return {"events": db.get_run_events(run_id, category=category)}


@app.get("/api/runs/{run_id}/export.csv")
def api_export_csv(run_id: int):
    if not db.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    csv_text = db.export_run_csv(run_id)
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="agri_run_{run_id}.csv"'},
    )


# ---------------- Image Asset Endpoints ----------------
@app.get("/api/images/dataset/{filename}")
def serve_dataset_image(filename: str):
    safe = os.path.basename(filename)
    path = os.path.join(db.DATASET_IMAGES_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/images/snapshots/{filename}")
def serve_snapshot_image(filename: str):
    safe = os.path.basename(filename)
    path = os.path.join(db.SNAPSHOTS_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(path, media_type="image/jpeg")


# ---------------- GStreamer Live Streaming Endpoints ----------------
@app.get("/api/gstreamer/status")
def api_gst_status():
    return gst_manager.get_all_status()


@app.post("/api/gstreamer/pipeline")
def api_gst_set_pipeline(req: GstPipelineReq):
    if req.channel not in gst_manager.channels:
        raise HTTPException(status_code=400, detail="Unknown channel")
    ok, msg = gst_manager.channels[req.channel].set_custom_pipeline(req.pipeline)
    if ros_bridge and ros_bridge.active_run_id:
        db.add_event(
            ros_bridge.active_run_id,
            "info" if ok else "warn",
            "gstreamer",
            f"[{req.channel.upper()}] {msg}",
        )
    return {"ok": ok, "message": msg, "status": gst_manager.channels[req.channel].status()}


@app.get("/api/stream/{camera}")
async def api_mjpeg_stream(camera: str):
    if camera not in gst_manager.channels:
        raise HTTPException(status_code=404, detail="Camera channel not found")

    channel = gst_manager.channels[camera]

    async def frame_generator():
        while True:
            jpeg = channel.get_jpeg()
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(0.066)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ---------------- Operator Robot Controls ----------------
@app.post("/api/control/cmd_vel")
def api_cmd_vel(req: CmdVelReq):
    return ros_bridge.send_cmd_vel(req.linear_x, req.angular_z, duration_s=req.duration_s)


@app.post("/api/control/estop")
def api_estop(req: EstopReq):
    res = ros_bridge.set_estop(req.active)
    target_run = ros_bridge.active_run_id
    if target_run:
        db.add_event(
            target_run,
            "alert" if req.active else "info",
            "mission",
            "EMERGENCY STOP ACTIVATED by operator." if req.active else "Emergency Stop released.",
        )
    return res


@app.post("/api/control/mast_height")
def api_mast_height(req: MastHeightReq):
    res = ros_bridge.set_mast_height(req.height_m, auto_align=req.auto_align)
    if ros_bridge.active_run_id:
        db.add_event(
            ros_bridge.active_run_id,
            "info",
            "mast",
            f"Operator commanded camera mast to {res['mast_target_m']:.3f} m (auto_align={res['auto_align_mast']}).",
        )
    return res


@app.post("/api/control/mission")
def api_mission_control(req: MissionReq):
    return ros_bridge.handle_mission_action(req.action, explicit_run_id=req.run_id)


@app.post("/api/control/snapshot")
def api_capture_snapshot(req: SnapshotReq):
    if req.camera not in gst_manager.channels:
        raise HTTPException(status_code=400, detail="Invalid camera")

    ch = gst_manager.channels[req.camera]
    if not ch.is_ros_live() and ch.mode != "custom_gst":
        raise HTTPException(
            status_code=503,
            detail="No live camera feed active yet. Launch Gazebo or a GStreamer source first.",
        )

    bgr = ch.get_bgr()
    if bgr is None:
        raise HTTPException(status_code=503, detail="No camera frame available yet")

    run_id = req.run_id
    if run_id is None or not db.get_run(run_id):
        run_id = int(db.ensure_active_run("Gazebo Live Trial")["id"])
        ros_bridge.active_run_id = run_id

    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    fname = f"snap_{req.camera}_{ts_tag}.jpg"
    out_path = os.path.join(db.SNAPSHOTS_DIR, fname)
    os.makedirs(db.SNAPSHOTS_DIR, exist_ok=True)
    cv2.imwrite(out_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    telem = ros_bridge.get_telemetry()
    x_m = float(telem["x_m"])
    y_m = float(telem["y_m"])
    mast_h = float(telem["mast_height_m"])

    rows = db.FIELD_GEOMETRY["rows"]
    nearest_row = min(rows, key=lambda r: abs(r["x"] - x_m))["index"]
    lanes = db.FIELD_GEOMETRY["lanes"]
    nearest_lane = min(lanes, key=lambda l: abs(l["x"] - x_m))["name"]

    det = db.add_detection(
        run_id=run_id,
        disease_class=req.disease_class,
        confidence=req.confidence,
        x_m=x_m,
        y_m=y_m,
        row_index=nearest_row,
        lane=nearest_lane,
        camera_side=req.camera if req.camera in ("left", "right") else "right",
        mast_height_m=mast_h,
        image_path=f"snapshots/{fname}",
    )
    db.add_event(
        run_id,
        "alert" if req.disease_class != "healthy" else "info",
        "detection",
        f"Manual snapshot logged ({req.disease_class}) on Row {nearest_row} at ({x_m:.2f}, {y_m:.2f}m).",
    )
    return det


# ---------------- Real-Time Telemetry WebSocket ----------------
@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = ros_bridge.get_telemetry()
            payload["gst"] = {
                k: {"fps": ch.fps, "mode": ch.status()["mode"], "ros_live": ch.is_ros_live()}
                for k, ch in gst_manager.channels.items()
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
    except Exception:
        return


# Serve built React production bundle if present
FRONTEND_DIST = os.path.join(os.path.dirname(db.BASE_DIR), "frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
