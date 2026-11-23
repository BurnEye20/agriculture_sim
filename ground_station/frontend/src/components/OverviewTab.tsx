import React from "react";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  AreaChart,
  Area,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  Compass,
  HeartPulse,
  MapPin,
  Ruler,
} from "lucide-react";
import { DISEASE_STYLES, type Detection, type RunStats, type Telemetry } from "../types";
import { imageUrl } from "../api";

interface Props {
  stats: RunStats | null;
  detections: Detection[];
  telemetry: Telemetry | null;
  onInspectDetection: (det: Detection) => void;
  onJumpToTab: (tab: "heatmap" | "detections" | "live") => void;
}

export const OverviewTab: React.FC<Props> = ({
  stats,
  detections,
  telemetry,
  onInspectDetection,
  onJumpToTab,
}) => {
  const totalDets = stats?.total_detections ?? 0;
  const diseasedCount = stats?.diseased_count ?? 0;
  const healthyCount = stats?.healthy_count ?? 0;
  const healthLabel =
    stats?.health_score_pct !== null && stats?.health_score_pct !== undefined
      ? `${stats.health_score_pct.toFixed(1)}%`
      : "—";
  const distanceM = stats?.run?.distance_m ?? 0;
  const maxDriftM = stats?.odom_drift?.max_drift_m ?? 0;
  const mastH = telemetry?.ros_connected
    ? telemetry.mast_height_m
    : stats?.run?.avg_mast_height_m ?? 0;

  const diseaseBreakdown = stats?.disease_breakdown ?? [
    { key: "anthracnose_fruit_rot", name: "Anthracnose", full_name: "Anthracnose Fruit Rot", count: 0, color: "#dc2626", pct: 0 },
    { key: "gray_mold", name: "Gray Mold", full_name: "Gray Mold (Botrytis)", count: 0, color: "#d97706", pct: 0 },
    { key: "powdery_mildew", name: "Powdery Mildew", full_name: "Powdery Mildew", count: 0, color: "#7c3aed", pct: 0 },
    { key: "healthy", name: "Healthy", full_name: "Healthy Fruit", count: 0, color: "#059669", pct: 0 },
  ];
  const rowBreakdown = stats?.row_breakdown ?? [
    { row: "Row 1", row_index: 1, healthy: 0, anthracnose_fruit_rot: 0, gray_mold: 0, powdery_mildew: 0, diseased_total: 0, total: 0 },
    { row: "Row 2", row_index: 2, healthy: 0, anthracnose_fruit_rot: 0, gray_mold: 0, powdery_mildew: 0, diseased_total: 0, total: 0 },
    { row: "Row 3", row_index: 3, healthy: 0, anthracnose_fruit_rot: 0, gray_mold: 0, powdery_mildew: 0, diseased_total: 0, total: 0 },
    { row: "Row 4", row_index: 4, healthy: 0, anthracnose_fruit_rot: 0, gray_mold: 0, powdery_mildew: 0, diseased_total: 0, total: 0 },
  ];
  const zoneProfile = stats?.zone_profile ?? [
    { zone: "0–5m", y_start: 0, diseased: 0, healthy: 0 },
    { zone: "5–10m", y_start: 5, diseased: 0, healthy: 0 },
    { zone: "10–15m", y_start: 10, diseased: 0, healthy: 0 },
    { zone: "15–20m", y_start: 15, diseased: 0, healthy: 0 },
    { zone: "20–25m", y_start: 20, diseased: 0, healthy: 0 },
    { zone: "25–30m", y_start: 25, diseased: 0, healthy: 0 },
    { zone: "30–35m", y_start: 30, diseased: 0, healthy: 0 },
  ];

  const recentDiseased = detections
    .filter((d) => d.disease_class !== "healthy")
    .slice(0, 6);

  return (
    <div className="space-y-6">
      {!stats && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-white px-5 py-3.5 text-xs text-slate-600">
          <span>
            Database is clean (<strong className="font-mono text-slate-900">0 runs · 0 captures</strong>). Launch Gazebo (<code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[11px]">agriculture_world.launch.xml</code>) to automatically record your first live trial.
          </span>
          <button
            onClick={() => onJumpToTab("live")}
            className="inline-flex items-center gap-1 rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-800"
          >
            Open Live & Control <ArrowUpRight className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* Top KPI Row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          label="Field Health Index"
          value={healthLabel}
          sub={`${healthyCount} healthy / ${totalDets} samples`}
          icon={<HeartPulse className="h-4 w-4 text-emerald-600" />}
          accent="emerald"
        />
        <KpiCard
          label="Disease Detections"
          value={String(diseasedCount)}
          sub={
            totalDets > 0
              ? `${stats?.reviewed_diseased ?? 0} reviewed · avg conf ${Math.round(
                  (stats?.run.avg_confidence ?? 0) * 100
                )}%`
              : "No detections recorded yet"
          }
          icon={<AlertTriangle className="h-4 w-4 text-red-600" />}
          accent="red"
        />
        <KpiCard
          label="Coverage & EKF Drift"
          value={`${distanceM.toFixed(1)} m`}
          sub={`Raw odom drift ${maxDriftM.toFixed(2)}m (EKF compensated)`}
          icon={<Compass className="h-4 w-4 text-sky-600" />}
          accent="sky"
        />
        <KpiCard
          label="Telescopic Mast Height"
          value={mastH > 0 ? `${mastH.toFixed(3)} m` : "—"}
          sub={
            telemetry?.ros_connected
              ? `Live · ${telemetry.auto_align_mast ? "Auto-Align ON" : "Manual"}`
              : "Waiting for rover telemetry"
          }
          icon={<Ruler className="h-4 w-4 text-slate-700" />}
          accent="slate"
        />
      </div>

      {/* Main Analytics Grid */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        {/* Disease Distribution Donut */}
        <div className="rounded-2xl border border-slate-200/90 bg-white p-5 lg:col-span-4">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">Disease Breakdown</h3>
              <p className="text-xs text-slate-500">YOLOv11 classifications across all rows</p>
            </div>
            <Activity className="h-4 w-4 text-slate-400" />
          </div>

          <div className="flex h-48 w-full items-center justify-center">
            {totalDets === 0 ? (
              <div className="flex h-36 w-36 items-center justify-center rounded-full border-8 border-slate-100 text-center font-mono text-xs text-slate-400">
                0 samples
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={diseaseBreakdown}
                    dataKey="count"
                    nameKey="name"
                    innerRadius={52}
                    outerRadius={76}
                    paddingAngle={3}
                    strokeWidth={0}
                  >
                    {diseaseBreakdown.map((entry) => (
                      <Cell key={entry.key} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      borderRadius: 12,
                      border: "1px solid #e2e8f0",
                      fontSize: 12,
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="mt-3 space-y-2">
            {diseaseBreakdown.map((item) => (
              <div
                key={item.key}
                className="flex items-center justify-between text-xs"
              >
                <div className="flex items-center gap-2 text-slate-700">
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span>{item.name}</span>
                </div>
                <div className="font-mono text-slate-600">
                  {item.count}{" "}
                  <span className="text-slate-400">({item.pct}%)</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Row-by-Row Stacked Bar Chart */}
        <div className="rounded-2xl border border-slate-200/90 bg-white p-5 lg:col-span-5">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Row-by-Row Disease Count
              </h3>
              <p className="text-xs text-slate-500">
                4 elevated hydroponic rows (±0.25° uneven grade)
              </p>
            </div>
            <button
              onClick={() => onJumpToTab("heatmap")}
              className="inline-flex items-center gap-1 text-xs font-medium text-emerald-700 hover:text-emerald-800"
            >
              Open Heatmap <ArrowUpRight className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rowBreakdown} barSize={26}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis
                  dataKey="row"
                  tick={{ fontSize: 12, fill: "#64748b" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  allowDecimals={false}
                  tick={{ fontSize: 11, fill: "#94a3b8" }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: 12,
                    border: "1px solid #e2e8f0",
                    fontSize: 12,
                  }}
                />
                <Bar
                  dataKey="anthracnose_fruit_rot"
                  name="Anthracnose"
                  stackId="a"
                  fill="#dc2626"
                />
                <Bar
                  dataKey="gray_mold"
                  name="Gray Mold"
                  stackId="a"
                  fill="#d97706"
                />
                <Bar
                  dataKey="powdery_mildew"
                  name="Powdery Mildew"
                  stackId="a"
                  fill="#7c3aed"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Longitudinal 0-35m Profile */}
        <div className="rounded-2xl border border-slate-200/90 bg-white p-5 lg:col-span-3">
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-slate-900">Aisle Depth Profile</h3>
            <p className="text-xs text-slate-500">Diseased hits along Y (0–35 m)</p>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={zoneProfile}>
                <defs>
                  <linearGradient id="disGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#dc2626" stopOpacity={0.28} />
                    <stop offset="95%" stopColor="#dc2626" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis
                  dataKey="zone"
                  tick={{ fontSize: 10, fill: "#64748b" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  allowDecimals={false}
                  tick={{ fontSize: 11, fill: "#94a3b8" }}
                  axisLine={false}
                  tickLine={false}
                  width={26}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: 12,
                    border: "1px solid #e2e8f0",
                    fontSize: 12,
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="diseased"
                  name="Diseased"
                  stroke="#dc2626"
                  strokeWidth={2}
                  fill="url(#disGrad)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recent Disease Captures Strip */}
      <div className="rounded-2xl border border-slate-200/90 bg-white p-5">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              Latest Disease Captures & Locations
            </h3>
            <p className="text-xs text-slate-500">
              Click any camera shot to inspect exact greenhouse coordinates & log treatment
            </p>
          </div>
          <button
            onClick={() => onJumpToTab("detections")}
            className="inline-flex items-center gap-1 text-xs font-medium text-emerald-700 hover:text-emerald-800"
          >
            View All Captures <ArrowUpRight className="h-3.5 w-3.5" />
          </button>
        </div>

        {recentDiseased.length === 0 ? (
          <div className="rounded-xl bg-slate-50 p-6 text-center text-xs text-slate-500">
            No diseased samples recorded in this run yet.
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {recentDiseased.map((det) => {
              const st = DISEASE_STYLES[det.disease_class] ?? DISEASE_STYLES.healthy;
              return (
                <button
                  key={det.id}
                  onClick={() => onInspectDetection(det)}
                  className="group flex items-center gap-3.5 rounded-xl border border-slate-200/80 bg-slate-50/40 p-2.5 text-left transition hover:border-slate-300 hover:bg-white hover:shadow-sm"
                >
                  <img
                    src={imageUrl(det.image_path)}
                    alt={st.short}
                    className="h-16 w-16 shrink-0 rounded-lg border border-slate-200 object-cover"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className={`inline-flex items-center rounded-md border px-1.5 py-0.5 text-[11px] font-medium ${st.bg} ${st.text} ${st.border}`}
                      >
                        {st.short}
                      </span>
                      <span className="font-mono text-[11px] text-slate-500">
                        {Math.round(det.confidence * 100)}%
                      </span>
                    </div>
                    <div className="mt-1.5 flex items-center gap-1 text-xs text-slate-700">
                      <MapPin className="h-3 w-3 text-slate-400" />
                      <span className="font-medium">Row {det.row_index}</span>
                      <span className="text-slate-300">·</span>
                      <span className="font-mono text-[11px] text-slate-500">
                        ({det.x_m.toFixed(2)}, {det.y_m.toFixed(2)}m)
                      </span>
                    </div>
                    <div className="mt-0.5 text-[11px] text-slate-400">
                      Mast {det.mast_height_m.toFixed(2)}m · {det.camera_side} cam
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

const KpiCard: React.FC<{
  label: string;
  value: string;
  sub: string;
  icon: React.ReactNode;
  accent: string;
}> = ({ label, value, sub, icon }) => (
  <div className="rounded-2xl border border-slate-200/90 bg-white p-4">
    <div className="flex items-center justify-between">
      <span className="text-xs font-medium text-slate-500">{label}</span>
      <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-50 border border-slate-100">
        {icon}
      </span>
    </div>
    <div className="mt-2 font-mono text-2xl font-semibold tracking-tight text-slate-900">
      {value}
    </div>
    <div className="mt-1 text-xs text-slate-500">{sub}</div>
  </div>
);
