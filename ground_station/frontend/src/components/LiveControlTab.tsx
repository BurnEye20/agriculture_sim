import React, { useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Camera,
  Columns2,
  OctagonAlert,
  Pause,
  Play,
  Radio,
  RotateCcw,
  Ruler,
  Settings2,
  Square,
  Video,
} from "lucide-react";
import { api, streamUrl } from "../api";
import type { GstStatusResponse, Telemetry } from "../types";

interface Props {
  activeRunId: number | null;
  telemetry: Telemetry | null;
  onRefreshData: () => void;
}

type CamLayout = "left" | "right" | "bottom" | "dual";

export const LiveControlTab: React.FC<Props> = ({
  activeRunId,
  telemetry,
  onRefreshData,
}) => {
  const [layout, setLayout] = useState<CamLayout>("dual");
  const [linearSpeed, setLinearSpeed] = useState(0.45);
  const [angularSpeed, setAngularSpeed] = useState(0.75);
  const [mastSlider, setMastSlider] = useState(0.94);
  const [autoAlign, setAutoAlign] = useState(true);
  const [gstStatus, setGstStatus] = useState<GstStatusResponse | null>(null);
  const [showPipelineCfg, setShowPipelineCfg] = useState(false);
  const [pipeChannel, setPipeChannel] = useState<"left" | "right" | "bottom">("left");
  const [pipeInput, setPipeInput] = useState("");
  const [snapClass, setSnapClass] = useState("gray_mold");
  const [feedback, setFeedback] = useState<{ text: string; warn?: boolean } | null>(null);
  const holdTimerRef = useRef<number | null>(null);
  const pressStartRef = useRef<number>(0);

  useEffect(() => {
    api.getGstStatus().then(setGstStatus).catch(() => {});
  }, []);

  useEffect(() => {
    if (telemetry) {
      setAutoAlign(telemetry.auto_align_mast);
      if (telemetry.auto_align_mast && telemetry.mast_height_m > 0.5) {
        setMastSlider(telemetry.mast_height_m);
      }
    }
  }, [telemetry?.auto_align_mast, telemetry?.mast_height_m]);

  const notify = (text: string, warn = false) => {
    setFeedback({ text, warn });
    window.setTimeout(() => setFeedback(null), 3400);
  };

  const sendDrivePulse = async (vx: number, wz: number, durationS = 0.55) => {
    try {
      const res = await api.sendCmdVel(vx, wz, durationS);
      if (!res.ok && res.reason) {
        notify(res.reason, true);
      }
    } catch {
      notify("Failed to reach Ground Station backend.", true);
    }
  };

  const startDrive = (vx: number, wz: number) => {
    stopDrive(false);
    pressStartRef.current = performance.now();
    void sendDrivePulse(vx, wz, 0.55);
    holdTimerRef.current = window.setInterval(() => {
      void sendDrivePulse(vx, wz, 0.55);
    }, 180);
  };

  const stopDrive = (sendZero = true) => {
    const heldMs = performance.now() - pressStartRef.current;
    if (holdTimerRef.current !== null) {
      window.clearInterval(holdTimerRef.current);
      holdTimerRef.current = null;
    }
    if (sendZero) {
      // On a quick click (<220ms), let the 0.45s ROS 2 watchdog finish the nudge so
      // a single click visibly moves the rover; when held longer, stop immediately on release.
      if (heldMs >= 220) {
        void api.sendCmdVel(0, 0, 0.1).catch(() => {});
      }
    }
  };

  const forceStop = () => {
    if (holdTimerRef.current !== null) {
      window.clearInterval(holdTimerRef.current);
      holdTimerRef.current = null;
    }
    void api.sendCmdVel(0, 0, 0.1).catch(() => {});
    notify("Rover stop command sent (0.00 m/s).");
  };

  // WASD keyboard teleop (when not typing in an input)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.repeat ||
        ["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement)?.tagName)
      ) {
        return;
      }
      const k = e.key.toLowerCase();
      if (k === "w") startDrive(linearSpeed, 0);
      else if (k === "s") startDrive(-linearSpeed, 0);
      else if (k === "a") startDrive(0, angularSpeed);
      else if (k === "d") startDrive(0, -angularSpeed);
      else if (k === " ") {
        e.preventDefault();
        forceStop();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (["w", "a", "s", "d"].includes(k)) stopDrive(true);
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      stopDrive(false);
    };
  }, [linearSpeed, angularSpeed]);

  const handleMastCommit = async (height: number, auto = false) => {
    setMastSlider(height);
    setAutoAlign(auto);
    const res = await api.setMastHeight(height, auto);
    if (!res.ros_connected) {
      notify(`Mast target ${height.toFixed(3)}m queued (ROS 2 offline)`, true);
    } else {
      notify(
        auto
          ? "Mast Auto-Align enabled (visual servoing active)"
          : `Mast commanded to ${height.toFixed(3)} m (manual)`
      );
    }
  };

  const handleApplyPipeline = async (preset?: string) => {
    const val = preset !== undefined ? preset : pipeInput;
    const res = await api.setGstPipeline(pipeChannel, val);
    notify(res.message, !res.ok);
    api.getGstStatus().then(setGstStatus).catch(() => {});
  };

  const handleSnapshot = async (cam: "left" | "right" | "bottom") => {
    try {
      const det = await api.captureSnapshot(activeRunId, cam, snapClass, 0.91);
      notify(`Saved live ${cam.toUpperCase()} snapshot to Run #${det.run_id}.`);
      onRefreshData();
    } catch (err) {
      notify(
        err instanceof Error ? err.message : "No live Gazebo camera feed to capture.",
        true
      );
    }
  };

  const estopActive = telemetry?.estop_active ?? false;

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
      {/* Left 8 Cols: Real-Time GStreamer Video Viewport */}
      <div className="space-y-4 xl:col-span-8">
        <div className="rounded-2xl border border-slate-200/90 bg-white p-4">
          {/* Stream Header & Channel Switcher */}
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-1 text-[11px] font-semibold text-emerald-700 border border-emerald-200">
                <Radio className="h-3 w-3 animate-pulse" />
                {gstStatus?.gst_version ?? "GStreamer 1.24"}
              </span>
              <span className="font-mono text-xs text-slate-500">
                {telemetry?.ros_connected ? "ROS 2 LIVE" : "AWAITING GAZEBO"}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-1.5">
              {(
                [
                  { k: "dual", label: "Dual Row", icon: <Columns2 className="h-3.5 w-3.5" /> },
                  { k: "left", label: "Left Cam" },
                  { k: "right", label: "Right Cam" },
                  { k: "bottom", label: "ArUco Cam" },
                ] as { k: CamLayout; label: string; icon?: React.ReactNode }[]
              ).map((item) => (
                <button
                  key={item.k}
                  onClick={() => setLayout(item.k)}
                  className={`inline-flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs font-medium transition ${
                    layout === item.k
                      ? "bg-slate-900 text-white"
                      : "text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  {item.icon}
                  {item.label}
                </button>
              ))}
              <button
                onClick={() => setShowPipelineCfg((v) => !v)}
                className={`inline-flex items-center gap-1 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition ${
                  showPipelineCfg
                    ? "border-slate-900 bg-slate-50 text-slate-900"
                    : "border-slate-200 text-slate-600 hover:bg-slate-50"
                }`}
              >
                <Settings2 className="h-3.5 w-3.5" />
                GStreamer
              </button>
            </div>
          </div>

          {/* Expandable GStreamer Custom Pipeline Configurator */}
          {showPipelineCfg && (
            <div className="mb-4 rounded-xl border border-slate-200 bg-slate-50 p-3.5 text-xs">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <span className="font-semibold text-slate-800">
                  GStreamer Source Pipeline (Real-World Deployment)
                </span>
                <div className="flex items-center gap-1">
                  {(["left", "right", "bottom"] as const).map((c) => (
                    <button
                      key={c}
                      onClick={() => setPipeChannel(c)}
                      className={`rounded-md px-2 py-0.5 text-[11px] font-medium uppercase ${
                        pipeChannel === c
                          ? "bg-slate-800 text-white"
                          : "bg-white text-slate-600 border border-slate-200"
                      }`}
                    >
                      {c}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex flex-col gap-2 sm:flex-row">
                <input
                  value={pipeInput}
                  onChange={(e) => setPipeInput(e.target.value)}
                  placeholder="e.g. udpsrc port=5000 ! application/x-rtp,encoding-name=VP8 ! rtpvp8depay ! vp8dec"
                  className="flex-1 rounded-lg border border-slate-200 bg-white px-3 py-1.5 font-mono text-[11px] text-slate-800 focus:border-emerald-500 focus:outline-none"
                />
                <button
                  onClick={() => handleApplyPipeline()}
                  className="rounded-lg bg-slate-900 px-3 py-1.5 font-medium text-white hover:bg-slate-800"
                >
                  Apply
                </button>
              </div>
              <div className="mt-2 flex flex-wrap gap-2">
                <button
                  onClick={() => {
                    setPipeInput("");
                    handleApplyPipeline("ros2");
                  }}
                  className="rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-600 hover:bg-slate-100"
                >
                  Preset: ROS 2 appsrc
                </button>
                <button
                  onClick={() => {
                    const p = "videotestsrc is-live=true pattern=ball";
                    setPipeInput(p);
                    handleApplyPipeline(p);
                  }}
                  className="rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-600 hover:bg-slate-100"
                >
                  Preset: GStreamer Test Pattern
                </button>
                <button
                  onClick={() =>
                    setPipeInput(
                      "udpsrc port=5000 ! application/x-rtp,encoding-name=VP8 ! rtpvp8depay ! vp8dec"
                    )
                  }
                  className="rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-600 hover:bg-slate-100"
                >
                  Template: UDP/RTP VP8
                </button>
              </div>
              {gstStatus?.channels[pipeChannel] && (
                <div className="mt-2 truncate font-mono text-[10px] text-slate-500">
                  Active: {gstStatus.channels[pipeChannel].pipeline}
                </div>
              )}
            </div>
          )}

          {/* Video Frames */}
          {layout === "dual" ? (
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              <StreamTile
                cam="left"
                title="Left Row Camera"
                fps={telemetry?.gst?.left?.fps ?? 10}
                onSnap={() => handleSnapshot("left")}
              />
              <StreamTile
                cam="right"
                title="Right Row Camera"
                fps={telemetry?.gst?.right?.fps ?? 10}
                onSnap={() => handleSnapshot("right")}
              />
            </div>
          ) : (
            <StreamTile
              cam={layout}
              title={
                layout === "left"
                  ? "Left Row Camera (YOLOv11)"
                  : layout === "right"
                  ? "Right Row Camera (YOLOv11)"
                  : "Bottom Tilted ArUco Camera (48.7°)"
              }
              fps={telemetry?.gst?.[layout]?.fps ?? 10}
              onSnap={() => handleSnapshot(layout)}
              large
            />
          )}

          {/* Snapshot Tag Bar */}
          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-3 text-xs">
            <div className="flex items-center gap-2 text-slate-600">
              <Camera className="h-3.5 w-3.5 text-slate-400" />
              <span>Manual capture tag:</span>
              <select
                value={snapClass}
                onChange={(e) => setSnapClass(e.target.value)}
                className="rounded-lg border border-slate-200 bg-slate-50 px-2 py-1 text-xs text-slate-800"
              >
                <option value="anthracnose_fruit_rot">Anthracnose</option>
                <option value="gray_mold">Gray Mold</option>
                <option value="powdery_mildew">Powdery Mildew</option>
                <option value="healthy">Healthy</option>
              </select>
            </div>
            {feedback && (
              <span
                className={`rounded-md px-2.5 py-1 text-[11px] font-medium ${
                  feedback.warn
                    ? "bg-amber-50 text-amber-800 border border-amber-200"
                    : "bg-emerald-50 text-emerald-700"
                }`}
              >
                {feedback.text}
              </span>
            )}
          </div>
        </div>

        {/* Live Telemetry Strip */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MiniStat
            label="EKF Pose (X, Y)"
            value={
              telemetry?.ros_connected
                ? `${telemetry.x_m.toFixed(2)}, ${telemetry.y_m.toFixed(2)} m`
                : "—"
            }
          />
          <MiniStat
            label="Heading / Vel"
            value={
              telemetry?.ros_connected
                ? `${telemetry.yaw_deg.toFixed(1)}° · ${telemetry.linear_vel.toFixed(2)} m/s`
                : "—"
            }
          />
          <MiniStat
            label="Last ArUco Lock"
            value={
              telemetry?.ros_connected &&
              telemetry?.last_aruco_id !== null &&
              telemetry?.last_aruco_id !== undefined
                ? `#${String(telemetry.last_aruco_id).padStart(2, "0")} (${telemetry.aruco_dist_m.toFixed(
                    2
                  )}m)`
                : "—"
            }
          />
          <MiniStat
            label="Odom Drift vs EKF"
            value={telemetry?.ros_connected ? `${telemetry.drift_m.toFixed(3)} m` : "—"}
          />
        </div>
      </div>

      {/* Right 4 Cols: Minimalistic Robot & Mast Control Deck */}
      <div className="space-y-4 xl:col-span-4">
        {/* Live Robot Status Indicator + Autonomous Mission + E-STOP */}
        <div className="rounded-2xl border border-slate-200/90 bg-white p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-900">Mission Control</h3>
            <RobotModeBadge mode={telemetry?.robot_mode ?? telemetry?.mission_state ?? "standby"} />
          </div>

          <div className="mb-3 rounded-xl border border-slate-100 bg-slate-50/80 px-3 py-2 text-xs">
            <div className="flex items-center justify-between text-[11px] text-slate-500">
              <span>Current Robot Status</span>
              {telemetry?.active_run_id ? (
                <span className="font-mono text-emerald-700">
                  Logging Run #{telemetry.active_run_id}
                </span>
              ) : (
                <span className="font-mono text-slate-400">No active run</span>
              )}
            </div>
            <div className="mt-0.5 font-medium text-slate-800">
              {telemetry?.status_detail ||
                (telemetry?.ros_connected ? "Connected · Ready" : "Waiting for Gazebo / ROS 2")}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2">
            <button
              onClick={async () => {
                const res = await api.sendMission("start", activeRunId ?? undefined);
                notify(res.message, !res.ros_connected);
                onRefreshData();
              }}
              className="inline-flex items-center justify-center gap-1.5 rounded-xl bg-emerald-600 px-3 py-2 text-xs font-semibold text-white transition hover:bg-emerald-700"
            >
              <Play className="h-3.5 w-3.5" /> Start
            </button>
            <button
              onClick={async () => {
                const res = await api.sendMission("pause", activeRunId ?? undefined);
                notify(res.message, !res.ros_connected);
              }}
              className="inline-flex items-center justify-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50"
            >
              <Pause className="h-3.5 w-3.5" /> Pause
            </button>
            <button
              onClick={async () => {
                const res = await api.sendMission("rtb", activeRunId ?? undefined);
                notify(res.message, !res.ros_connected);
              }}
              className="inline-flex items-center justify-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50"
            >
              <RotateCcw className="h-3.5 w-3.5" /> RTB
            </button>
          </div>

          <button
            onClick={async () => {
              await api.setEstop(!estopActive);
              notify(!estopActive ? "EMERGENCY STOP ENGAGED" : "E-STOP released");
            }}
            className={`mt-3 flex w-full items-center justify-center gap-2 rounded-xl py-2.5 text-xs font-bold tracking-wide transition ${
              estopActive
                ? "bg-amber-500 text-white hover:bg-amber-600"
                : "bg-red-600 text-white hover:bg-red-700"
            }`}
          >
            <OctagonAlert className="h-4 w-4" />
            {estopActive ? "RELEASE E-STOP" : "EMERGENCY STOP"}
          </button>
        </div>

        {/* Manual Teleoperation Pad */}
        <div className="rounded-2xl border border-slate-200/90 bg-white p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-900">Manual Teleop</h3>
            <span className="font-mono text-[11px] text-slate-400">WASD + Space</span>
          </div>

          <div className="mx-auto grid w-44 grid-cols-3 gap-2">
            <div />
            <DriveBtn
              icon={<ArrowUp className="h-4 w-4" />}
              onDown={() => startDrive(linearSpeed, 0)}
              onUp={() => stopDrive(true)}
            />
            <div />
            <DriveBtn
              icon={<ArrowLeft className="h-4 w-4" />}
              onDown={() => startDrive(0, angularSpeed)}
              onUp={() => stopDrive(true)}
            />
            <button
              onClick={forceStop}
              className="flex h-12 items-center justify-center rounded-xl border border-slate-200 bg-slate-50 text-slate-600 hover:bg-slate-100"
              title="Stop (Space)"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
            </button>
            <DriveBtn
              icon={<ArrowRight className="h-4 w-4" />}
              onDown={() => startDrive(0, -angularSpeed)}
              onUp={() => stopDrive(true)}
            />
            <div />
            <DriveBtn
              icon={<ArrowDown className="h-4 w-4" />}
              onDown={() => startDrive(-linearSpeed, 0)}
              onUp={() => stopDrive(true)}
            />
            <div />
          </div>

          <div className="mt-4 space-y-2.5 text-xs">
            <div className="flex items-center justify-between text-slate-600">
              <span>Linear Speed</span>
              <span className="font-mono">{linearSpeed.toFixed(2)} m/s</span>
            </div>
            <input
              type="range"
              min={0.15}
              max={0.65}
              step={0.05}
              value={linearSpeed}
              onChange={(e) => setLinearSpeed(parseFloat(e.target.value))}
              className="w-full accent-emerald-600"
            />

            <div className="flex items-center justify-between text-slate-600">
              <span>Turn Rate</span>
              <span className="font-mono">{angularSpeed.toFixed(2)} rad/s</span>
            </div>
            <input
              type="range"
              min={0.3}
              max={1.1}
              step={0.05}
              value={angularSpeed}
              onChange={(e) => setAngularSpeed(parseFloat(e.target.value))}
              className="w-full accent-emerald-600"
            />
          </div>
        </div>

        {/* Telescopic Camera Mast Control */}
        <div className="rounded-2xl border border-slate-200/90 bg-white p-4">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Ruler className="h-4 w-4 text-slate-500" />
              <h3 className="text-sm font-semibold text-slate-900">Camera Mast</h3>
            </div>
            <button
              onClick={() => handleMastCommit(mastSlider, !autoAlign)}
              className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold transition ${
                autoAlign
                  ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                  : "bg-slate-100 text-slate-600"
              }`}
            >
              {autoAlign ? "Auto-Align ON" : "Manual Mode"}
            </button>
          </div>
          <p className="mb-3 text-[11px] text-slate-500">
            3-section telescopic actuator (0.726 m – 1.105 m optical range)
          </p>

          <div className="mb-2 flex items-baseline justify-between font-mono text-xs">
            <span className="text-slate-500">
              Measured:{" "}
              <strong className="text-slate-900">
                {(telemetry?.mast_height_m ?? mastSlider).toFixed(3)} m
              </strong>
            </span>
            <span className="text-emerald-700">Target: {mastSlider.toFixed(3)} m</span>
          </div>

          <input
            type="range"
            min={0.726}
            max={1.105}
            step={0.005}
            value={mastSlider}
            onChange={(e) => setMastSlider(parseFloat(e.target.value))}
            onMouseUp={() => handleMastCommit(mastSlider, false)}
            onTouchEnd={() => handleMastCommit(mastSlider, false)}
            className="w-full accent-slate-900"
          />
          <div className="mt-2 grid grid-cols-3 gap-1.5 text-[11px]">
            {[
              { label: "Low 0.80m", val: 0.8 },
              { label: "Mid 0.94m", val: 0.94 },
              { label: "High 1.08m", val: 1.08 },
            ].map((p) => (
              <button
                key={p.label}
                onClick={() => handleMastCommit(p.val, false)}
                className="rounded-lg border border-slate-200 py-1 font-mono text-slate-600 hover:bg-slate-50"
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const StreamTile: React.FC<{
  cam: "left" | "right" | "bottom";
  title: string;
  fps: number;
  onSnap: () => void;
  large?: boolean;
}> = ({ cam, title, fps, onSnap, large }) => (
  <div className="group relative overflow-hidden rounded-xl border border-slate-200 bg-slate-950">
    <img
      src={streamUrl(cam)}
      alt={title}
      className={`w-full object-cover ${large ? "aspect-[16/10]" : "aspect-[4/3]"}`}
    />
    <div className="absolute bottom-2.5 left-2.5 right-2.5 flex items-center justify-between">
      <span className="inline-flex items-center gap-1.5 rounded-lg bg-slate-950/75 px-2.5 py-1 text-[11px] font-medium text-white backdrop-blur-sm">
        <Video className="h-3 w-3 text-emerald-400" />
        {title}
        <span className="font-mono text-slate-300">· {fps.toFixed(0)} fps</span>
      </span>
      <button
        onClick={onSnap}
        className="inline-flex items-center gap-1 rounded-lg bg-white/90 px-2.5 py-1 text-[11px] font-semibold text-slate-900 shadow-sm backdrop-blur-sm transition hover:bg-white"
      >
        <Camera className="h-3 w-3" /> Capture
      </button>
    </div>
  </div>
);

const DriveBtn: React.FC<{
  icon: React.ReactNode;
  onDown: () => void;
  onUp: () => void;
}> = ({ icon, onDown, onUp }) => (
  <button
    type="button"
    onPointerDown={(e) => {
      e.preventDefault();
      (e.currentTarget as HTMLButtonElement).setPointerCapture(e.pointerId);
      onDown();
    }}
    onPointerUp={() => onUp()}
    onPointerCancel={() => onUp()}
    className="flex h-12 select-none touch-none items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-700 shadow-2xs transition active:scale-95 active:bg-slate-900 active:text-white hover:bg-slate-50"
  >
    {icon}
  </button>
);

const MiniStat: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="rounded-xl border border-slate-200/90 bg-white px-3.5 py-2.5">
    <div className="text-[11px] text-slate-500">{label}</div>
    <div className="mt-0.5 font-mono text-xs font-semibold text-slate-900">{value}</div>
  </div>
);

const RobotModeBadge: React.FC<{ mode: string }> = ({ mode }) => {
  const m = mode.toLowerCase();
  const cfg =
    m === "running_mission" || m === "running"
      ? { label: "Running Mission", cls: "bg-emerald-50 text-emerald-700 border-emerald-200" }
      : m === "returning_to_base"
      ? { label: "Returning to Base", cls: "bg-sky-50 text-sky-700 border-sky-200" }
      : m === "manual_mode" || m === "paused"
      ? { label: "Manual Mode", cls: "bg-amber-50 text-amber-700 border-amber-200" }
      : m === "at_base" || m === "completed"
      ? { label: "At Base · Ready", cls: "bg-slate-100 text-slate-700 border-slate-200" }
      : m === "estop"
      ? { label: "E-STOP", cls: "bg-red-100 text-red-700 border-red-200" }
      : { label: "Standby", cls: "bg-slate-100 text-slate-500 border-slate-200" };
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${cfg.cls}`}
    >
      {cfg.label}
    </span>
  );
};
