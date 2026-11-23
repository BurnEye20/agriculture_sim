import React, { useMemo, useState } from "react";
import {
  CheckCircle2,
  Eye,
  Flame,
  Layers,
  MapPin,
  Navigation,
  QrCode,
  Ruler,
} from "lucide-react";
import {
  DISEASE_STYLES,
  type Detection,
  type HeatmapPayload,
  type Telemetry,
} from "../types";
import { imageUrl } from "../api";

interface Props {
  heatmap: HeatmapPayload | null;
  telemetry: Telemetry | null;
  onUpdateDetection: (id: number, reviewed: boolean, note?: string) => Promise<void>;
}

export const HeatmapTab: React.FC<Props> = ({
  heatmap,
  telemetry,
  onUpdateDetection,
}) => {
  const [diseaseFilter, setDiseaseFilter] = useState<string>("diseased");
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showEkfPath, setShowEkfPath] = useState(true);
  const [showRawOdom, setShowRawOdom] = useState(true);
  const [showAruco, setShowAruco] = useState(true);
  const [selected, setSelected] = useState<Detection | null>(null);

  // Horizontal layout: Y axis runs left-to-right (0..35m), X axis runs top-to-bottom (-1.2..4.0m)
  // This makes the 35m x 5.2m greenhouse map wide, ergonomic, and super readable on desktop screens!
  const SVG_W = 1040;
  const SVG_H = 440;
  const PAD_L = 58;
  const PAD_R = 28;
  const PAD_T = 36;
  const PAD_B = 42;

  const field = heatmap?.field;
  const yMin = field?.y_min ?? -2.2;
  const yMax = field?.y_max ?? 35.0;
  const xMin = field?.x_min ?? -1.2;
  const xMax = field?.x_max ?? 4.0;

  const toSvgX = (yMeters: number) =>
    PAD_L + ((yMeters - yMin) / (yMax - yMin)) * (SVG_W - PAD_L - PAD_R);
  const toSvgY = (xMeters: number) =>
    PAD_T + ((xMeters - xMin) / (xMax - xMin)) * (SVG_H - PAD_T - PAD_B);

  const visibleDetections = useMemo(() => {
    if (!heatmap) return [];
    return heatmap.detections.filter((d) => {
      if (diseaseFilter === "all") return true;
      if (diseaseFilter === "diseased") return d.disease_class !== "healthy";
      return d.disease_class === diseaseFilter;
    });
  }, [heatmap, diseaseFilter]);

  const diseasedForHeat = useMemo(
    () => visibleDetections.filter((d) => d.disease_class !== "healthy"),
    [visibleDetections]
  );

  // Build polyline strings
  const ekfPolyline = useMemo(() => {
    if (!heatmap) return "";
    return heatmap.ekf_path
      .map((p) => `${toSvgX(p.y).toFixed(1)},${toSvgY(p.x).toFixed(1)}`)
      .join(" ");
  }, [heatmap]);

  const rawPolyline = useMemo(() => {
    if (!heatmap) return "";
    return heatmap.raw_odom_path
      .map((p) => `${toSvgX(p.y).toFixed(1)},${toSvgY(p.x).toFixed(1)}`)
      .join(" ");
  }, [heatmap]);

  if (!heatmap || !field) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-12 text-center text-sm text-slate-500">
        Loading greenhouse spatial map…
      </div>
    );
  }

  const activeDet =
    selected ?? diseasedForHeat[0] ?? visibleDetections[0] ?? null;
  const activeStyle = activeDet
    ? DISEASE_STYLES[activeDet.disease_class] ?? DISEASE_STYLES.healthy
    : null;

  return (
    <div className="space-y-5">
      {/* Compact Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200/90 bg-white px-4 py-3">
        <div className="flex flex-wrap items-center gap-1.5">
          {[
            { k: "diseased", label: "All Diseases" },
            { k: "anthracnose_fruit_rot", label: "Anthracnose" },
            { k: "gray_mold", label: "Gray Mold" },
            { k: "powdery_mildew", label: "Powdery Mildew" },
            { k: "all", label: "Include Healthy" },
          ].map((opt) => (
            <button
              key={opt.k}
              onClick={() => setDiseaseFilter(opt.k)}
              className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition ${
                diseaseFilter === opt.k
                  ? "bg-slate-900 text-white"
                  : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          <ToggleChip
            active={showHeatmap}
            onClick={() => setShowHeatmap((v) => !v)}
            icon={<Flame className="h-3.5 w-3.5 text-red-500" />}
            label="Heatmap"
          />
          <ToggleChip
            active={showEkfPath}
            onClick={() => setShowEkfPath((v) => !v)}
            icon={<Navigation className="h-3.5 w-3.5 text-emerald-600" />}
            label="EKF Path"
          />
          <ToggleChip
            active={showRawOdom}
            onClick={() => setShowRawOdom((v) => !v)}
            icon={<Layers className="h-3.5 w-3.5 text-amber-500" />}
            label="Raw Odom Drift"
          />
          <ToggleChip
            active={showAruco}
            onClick={() => setShowAruco((v) => !v)}
            icon={<QrCode className="h-3.5 w-3.5 text-slate-600" />}
            label="24 ArUco"
          />
        </div>
      </div>

      {/* Map + Side Inspector */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
        <div className="rounded-2xl border border-slate-200/90 bg-white p-4 xl:col-span-8">
          <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Greenhouse Spatial Disease Heatmap
              </h3>
              <p className="text-xs text-slate-500">
                35.0 m × 5.2 m field · 2.0° ground slope · {diseasedForHeat.length} diseased hotspot
                {diseasedForHeat.length === 1 ? "" : "s"} plotted
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-600">
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-red-600" /> Anthracnose (
                {heatmap.detections.filter((d) => d.disease_class === "anthracnose_fruit_rot").length})
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-amber-600" /> Gray Mold (
                {heatmap.detections.filter((d) => d.disease_class === "gray_mold").length})
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-violet-600" /> Powdery Mildew (
                {heatmap.detections.filter((d) => d.disease_class === "powdery_mildew").length})
              </span>
              <span className="inline-flex items-center gap-1.5 text-slate-400">
                <span className="h-1.5 w-3.5 rounded-full bg-emerald-500" /> EKF
              </span>
              <span className="inline-flex items-center gap-1.5 text-slate-400">
                <span className="h-0.5 w-3.5 border-t-2 border-dashed border-amber-500" /> Raw
              </span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <svg
              viewBox={`0 0 ${SVG_W} ${SVG_H}`}
              className="w-full min-w-[740px] rounded-xl border border-slate-100 bg-[#FAFCFB]"
            >
              <defs>
                <clipPath id="mapPlotClip">
                  <rect
                    x={PAD_L - 8}
                    y={PAD_T - 20}
                    width={SVG_W - PAD_L - PAD_R + 16}
                    height={SVG_H - PAD_T - PAD_B + 32}
                    rx={8}
                  />
                </clipPath>
                <radialGradient id="heatRed" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#dc2626" stopOpacity="0.68" />
                  <stop offset="52%" stopColor="#ef4444" stopOpacity="0.30" />
                  <stop offset="100%" stopColor="#dc2626" stopOpacity="0" />
                </radialGradient>
                <radialGradient id="heatAmber" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#d97706" stopOpacity="0.65" />
                  <stop offset="55%" stopColor="#f59e0b" stopOpacity="0.28" />
                  <stop offset="100%" stopColor="#d97706" stopOpacity="0" />
                </radialGradient>
                <radialGradient id="heatViolet" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#7c3aed" stopOpacity="0.62" />
                  <stop offset="55%" stopColor="#8b5cf6" stopOpacity="0.26" />
                  <stop offset="100%" stopColor="#7c3aed" stopOpacity="0" />
                </radialGradient>
              </defs>

              {/* Vertical distance grid lines every 5m */}
              {[0, 5, 10, 15, 20, 25, 30, 35].map((ym) => {
                const sx = toSvgX(ym);
                return (
                  <g key={ym}>
                    <line
                      x1={sx}
                      y1={PAD_T}
                      x2={sx}
                      y2={SVG_H - PAD_B}
                      stroke="#e2e8f0"
                      strokeDasharray="3 4"
                    />
                    <text
                      x={sx}
                      y={SVG_H - 16}
                      textAnchor="middle"
                      fontSize="11"
                      fill="#94a3b8"
                      className="font-mono"
                    >
                      {ym}m
                    </text>
                  </g>
                );
              })}

              {/* 3 Rover Lanes */}
              {field.lanes.map((lane) => {
                const sy = toSvgY(lane.x);
                return (
                  <g key={lane.name}>
                    <line
                      x1={PAD_L}
                      y1={sy}
                      x2={SVG_W - PAD_R}
                      y2={sy}
                      stroke="#cbd5e1"
                      strokeWidth="1"
                      strokeDasharray="6 6"
                    />
                    <text
                      x={PAD_L - 8}
                      y={sy + 4}
                      textAnchor="end"
                      fontSize="10"
                      fill="#64748b"
                    >
                      {lane.name.toUpperCase()}
                    </text>
                  </g>
                );
              })}

              {/* 4 Elevated Strawberry Rows */}
              {field.rows.map((row) => {
                const sy = toSvgY(row.x);
                const x1 = toSvgX(field.row_y_min);
                const x2 = toSvgX(field.row_y_max);
                return (
                  <g key={row.index}>
                    <rect
                      x={x1}
                      y={sy - 11}
                      width={x2 - x1}
                      height={22}
                      rx={6}
                      fill="#ecfdf5"
                      stroke="#a7f3d0"
                      strokeWidth="1.2"
                    />
                    <text
                      x={PAD_L - 8}
                      y={sy + 4}
                      textAnchor="end"
                      fontSize="11"
                      fontWeight="600"
                      fill="#047857"
                    >
                      R{row.index}
                    </text>
                    <text
                      x={x2 + 5}
                      y={sy + 3}
                      fontSize="9"
                      fill="#94a3b8"
                      className="font-mono"
                    >
                      {row.slope_deg > 0 ? "+0.25°" : "-0.25°"}
                    </text>
                  </g>
                );
              })}

              {/* 24 Ground ArUco Landmarks */}
              {showAruco &&
                heatmap.aruco_landmarks.map((m) => (
                  <g key={m.id}>
                    <rect
                      x={toSvgX(m.y) - 4}
                      y={toSvgY(m.x) - 4}
                      width={8}
                      height={8}
                      rx={1.5}
                      fill="#ffffff"
                      stroke="#475569"
                      strokeWidth="1.2"
                    />
                    <rect
                      x={toSvgX(m.y) - 1.5}
                      y={toSvgY(m.x) - 1.5}
                      width={3}
                      height={3}
                      fill="#0f172a"
                    />
                  </g>
                ))}

              {/* Drifting Raw Odometry Path (Orange dashed, clipped cleanly inside map) */}
              {showRawOdom && rawPolyline && (
                <polyline
                  clipPath="url(#mapPlotClip)"
                  points={rawPolyline}
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth="2"
                  strokeDasharray="6 4"
                  strokeOpacity="0.85"
                />
              )}

              {/* EKF + ArUco Corrected Path (Green solid) */}
              {showEkfPath && ekfPolyline && (
                <polyline
                  clipPath="url(#mapPlotClip)"
                  points={ekfPolyline}
                  fill="none"
                  stroke="#059669"
                  strokeWidth="2.4"
                  strokeOpacity="0.9"
                />
              )}

              {/* Gaussian Disease Intensity Heatmap Blobs */}
              {showHeatmap &&
                diseasedForHeat.map((d) => {
                  const grad =
                    d.disease_class === "anthracnose_fruit_rot"
                      ? "url(#heatRed)"
                      : d.disease_class === "gray_mold"
                      ? "url(#heatAmber)"
                      : "url(#heatViolet)";
                  return (
                    <circle
                      key={`heat-${d.id}`}
                      cx={toSvgX(d.y_m)}
                      cy={toSvgY(d.x_m)}
                      r={28 + d.confidence * 16}
                      fill={grad}
                      className="pointer-events-none"
                    />
                  );
                })}

              {/* Clickable Detection Markers */}
              {visibleDetections.map((d) => {
                const st = DISEASE_STYLES[d.disease_class] ?? DISEASE_STYLES.healthy;
                const isSel = activeDet?.id === d.id;
                const isHealthy = d.disease_class === "healthy";
                return (
                  <g
                    key={d.id}
                    onClick={() => setSelected(d)}
                    className="cursor-pointer"
                  >
                    {isSel && (
                      <circle
                        cx={toSvgX(d.y_m)}
                        cy={toSvgY(d.x_m)}
                        r={12}
                        fill="none"
                        stroke="#0f172a"
                        strokeWidth="2.2"
                      />
                    )}
                    <circle
                      cx={toSvgX(d.y_m)}
                      cy={toSvgY(d.x_m)}
                      r={isHealthy ? 3.5 : isSel ? 7.5 : 6.5}
                      fill={st.color}
                      stroke="#ffffff"
                      strokeWidth={isHealthy ? "1.4" : "2"}
                    />
                  </g>
                );
              })}

              {/* Live Rover Position Marker (shown when Gazebo/ROS 2 is connected) */}
              {telemetry && telemetry.ros_connected && (
                <g
                  transform={`translate(${toSvgX(telemetry.y_m)}, ${toSvgY(
                    telemetry.x_m
                  )})`}
                >
                  <circle r="13" fill="#0ea5e9" fillOpacity="0.16" />
                  <circle
                    r="6"
                    fill="#0284c7"
                    stroke="#ffffff"
                    strokeWidth="2"
                  />
                  <text
                    x="10"
                    y="-8"
                    fontSize="10"
                    fontWeight="600"
                    fill="#0369a1"
                    className="font-mono"
                  >
                    ROVER ({telemetry.x_m.toFixed(2)}, {telemetry.y_m.toFixed(2)})
                  </text>
                </g>
              )}
            </svg>
          </div>
        </div>

        {/* Side Capture & Location Inspector */}
        <div className="rounded-2xl border border-slate-200/90 bg-white p-5 xl:col-span-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-900">
              Location & Shot Inspector
            </h3>
            <Eye className="h-4 w-4 text-slate-400" />
          </div>

          {activeDet && activeStyle ? (
            <div className="space-y-4">
              <div className="relative overflow-hidden rounded-xl border border-slate-200 bg-slate-100">
                <img
                  src={imageUrl(activeDet.image_path)}
                  alt={activeStyle.label}
                  className="aspect-[4/3] w-full object-cover"
                />
                <span
                  className={`absolute left-3 top-3 rounded-lg border px-2 py-0.5 text-xs font-semibold shadow-sm ${activeStyle.bg} ${activeStyle.text} ${activeStyle.border}`}
                >
                  {activeStyle.label}
                </span>
                <span className="absolute right-3 top-3 rounded-lg bg-slate-900/80 px-2 py-0.5 font-mono text-xs text-white">
                  {Math.round(activeDet.confidence * 100)}% conf
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2.5 text-xs">
                <InfoTile
                  icon={<MapPin className="h-3.5 w-3.5 text-slate-400" />}
                  label="Map Position"
                  value={`X ${activeDet.x_m.toFixed(2)}m · Y ${activeDet.y_m.toFixed(2)}m`}
                />
                <InfoTile
                  icon={<Layers className="h-3.5 w-3.5 text-slate-400" />}
                  label="Row & Lane"
                  value={`Row ${activeDet.row_index} (${activeDet.lane})`}
                />
                <InfoTile
                  icon={<Ruler className="h-3.5 w-3.5 text-slate-400" />}
                  label="Mast Height"
                  value={`${activeDet.mast_height_m.toFixed(3)} m`}
                />
                <InfoTile
                  icon={<Eye className="h-3.5 w-3.5 text-slate-400" />}
                  label="Camera Side"
                  value={`${activeDet.camera_side.toUpperCase()} CAM`}
                />
              </div>

              <div className="flex items-center justify-between border-t border-slate-100 pt-3">
                <span className="text-xs text-slate-500">
                  {activeDet.reviewed ? "Marked as reviewed" : "Pending agronomy review"}
                </span>
                <button
                  onClick={async () => {
                    await onUpdateDetection(activeDet.id, !activeDet.reviewed);
                    setSelected({ ...activeDet, reviewed: !activeDet.reviewed });
                  }}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                    activeDet.reviewed
                      ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                      : "bg-slate-900 text-white hover:bg-slate-800"
                  }`}
                >
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  {activeDet.reviewed ? "Reviewed" : "Mark Reviewed"}
                </button>
              </div>
            </div>
          ) : (
            <div className="rounded-xl bg-slate-50 p-8 text-center text-xs text-slate-500">
              Select a detection pin on the heatmap to inspect its camera capture and exact
              coordinates.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const ToggleChip: React.FC<{
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}> = ({ active, onClick, icon, label }) => (
  <button
    onClick={onClick}
    className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-medium transition ${
      active
        ? "border-slate-300 bg-slate-50 text-slate-800"
        : "border-slate-200 bg-white text-slate-400"
    }`}
  >
    {icon}
    {label}
  </button>
);

const InfoTile: React.FC<{ icon: React.ReactNode; label: string; value: string }> = ({
  icon,
  label,
  value,
}) => (
  <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-2.5">
    <div className="flex items-center gap-1 text-[11px] text-slate-500">
      {icon}
      {label}
    </div>
    <div className="mt-1 font-mono text-xs font-medium text-slate-900">{value}</div>
  </div>
);
