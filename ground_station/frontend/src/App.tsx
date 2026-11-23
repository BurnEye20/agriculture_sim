import React, { useCallback, useEffect, useState } from "react";
import {
  BarChart3,
  BatteryCharging,
  Camera,
  Database,
  Flame,
  OctagonAlert,
  Radio,
  Sprout,
  Video,
} from "lucide-react";
import { api, telemetryWsUrl } from "./api";
import type {
  Detection,
  HeatmapPayload,
  MissionEvent,
  Run,
  RunStats,
  TabKey,
  Telemetry,
} from "./types";
import { OverviewTab } from "./components/OverviewTab";
import { HeatmapTab } from "./components/HeatmapTab";
import { LiveControlTab } from "./components/LiveControlTab";
import { DetectionsTab } from "./components/DetectionsTab";
import { HistoryLogsTab } from "./components/HistoryLogsTab";

const TABS: { key: TabKey; label: string; icon: React.ReactNode }[] = [
  { key: "overview", label: "Overview", icon: <BarChart3 className="h-4 w-4" /> },
  { key: "heatmap", label: "Field Heatmap", icon: <Flame className="h-4 w-4" /> },
  { key: "live", label: "Live & Control", icon: <Video className="h-4 w-4" /> },
  { key: "detections", label: "Captures", icon: <Camera className="h-4 w-4" /> },
  { key: "history", label: "Runs & Logs", icon: <Database className="h-4 w-4" /> },
];

export default function App() {
  const [tab, setTab] = useState<TabKey>("overview");
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [stats, setStats] = useState<RunStats | null>(null);
  const [heatmap, setHeatmap] = useState<HeatmapPayload | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [events, setEvents] = useState<MissionEvent[]>([]);
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [inspectDet, setInspectDet] = useState<Detection | null>(null);

  const loadRuns = useCallback(async () => {
    const res = await api.listRuns();
    setRuns(res.runs);
    setSelectedRunId((prev) => {
      if (prev && res.runs.some((r) => r.id === prev)) return prev;
      return res.runs[0]?.id ?? null;
    });
  }, []);

  const loadRunData = useCallback(async (runId: number) => {
    const [st, hm, dets, logs] = await Promise.all([
      api.getRunStats(runId),
      api.getRunHeatmap(runId),
      api.getRunDetections(runId),
      api.getRunLogs(runId),
    ]);
    setStats(st);
    setHeatmap(hm);
    setDetections(dets.detections);
    setEvents(logs.events);
  }, []);

  useEffect(() => {
    loadRuns().catch(() => {});
  }, [loadRuns]);

  useEffect(() => {
    if (selectedRunId !== null) {
      loadRunData(selectedRunId).catch(() => {});
    } else {
      setStats(null);
      setDetections([]);
      setEvents([]);
      api.getEmptyHeatmap().then(setHeatmap).catch(() => {});
    }
  }, [selectedRunId, loadRunData]);

  // Auto-switch UI to newly started live run (e.g. Run #2 created after RTB + Start)
  const prevActiveRunRef = React.useRef<number | null>(null);
  useEffect(() => {
    const liveActive = telemetry?.active_run_id ?? null;
    if (liveActive !== null && liveActive !== prevActiveRunRef.current) {
      prevActiveRunRef.current = liveActive;
      loadRuns()
        .then(() => setSelectedRunId(liveActive))
        .catch(() => {});
    }
  }, [telemetry?.active_run_id, loadRuns]);

  // Auto-refresh live Gazebo trial data every 2.5 seconds while connected
  useEffect(() => {
    if (!telemetry?.ros_connected) return;
    const id = window.setInterval(() => {
      loadRuns().catch(() => {});
      const targetId = selectedRunId ?? telemetry.active_run_id ?? null;
      if (targetId !== null) {
        loadRunData(targetId).catch(() => {});
      }
    }, 2500);
    return () => window.clearInterval(id);
  }, [telemetry?.ros_connected, telemetry?.active_run_id, selectedRunId, loadRuns, loadRunData]);

  // Real-time Telemetry WebSocket with auto-reconnect
  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: number | null = null;
    let unmounted = false;

    const connect = () => {
      if (unmounted) return;
      ws = new WebSocket(telemetryWsUrl());
      ws.onopen = () => setWsConnected(true);
      ws.onmessage = (ev) => {
        try {
          setTelemetry(JSON.parse(ev.data) as Telemetry);
        } catch {
          // ignore malformed packet
        }
      };
      ws.onclose = () => {
        setWsConnected(false);
        if (!unmounted) {
          timer = window.setTimeout(connect, 2000);
        }
      };
    };

    connect();
    return () => {
      unmounted = true;
      if (timer) window.clearTimeout(timer);
      ws?.close();
    };
  }, []);

  const handleUpdateDetection = async (
    id: number,
    reviewed: boolean,
    note?: string
  ) => {
    await api.updateDetection(id, { reviewed, operator_note: note });
    if (selectedRunId !== null) {
      await loadRunData(selectedRunId);
    }
  };

  const handleCreateRun = async (name: string, notes: string) => {
    const created = await api.createRun(name, notes);
    await loadRuns();
    setSelectedRunId(created.id);
  };

  const handleCompleteRun = async (runId: number) => {
    await api.completeRun(runId, "completed");
    await loadRuns();
    await loadRunData(runId);
  };

  const handleDeleteRun = async (runId: number) => {
    await api.deleteRun(runId);
    await loadRuns();
  };

  const handleResetDatabase = async () => {
    await api.resetDatabase();
    setSelectedRunId(null);
    setStats(null);
    setDetections([]);
    setEvents([]);
    await loadRuns();
    api.getEmptyHeatmap().then(setHeatmap).catch(() => {});
  };

  const estopActive = telemetry?.estop_active ?? false;

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-900">
      {/* Minimalistic Sticky Top Header */}
      <header className="sticky top-0 z-30 border-b border-slate-200/80 bg-white/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          {/* Brand */}
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-emerald-600 text-white shadow-xs">
              <Sprout className="h-4 w-4" />
            </div>
            <div>
              <div className="text-sm font-semibold tracking-tight text-slate-900">
                AgriRover <span className="font-normal text-slate-400">Ground Station</span>
              </div>
              <div className="flex items-center gap-2 text-[11px] text-slate-500">
                <span className="inline-flex items-center gap-1">
                  <Radio
                    className={`h-2.5 w-2.5 ${
                      telemetry?.ros_connected
                        ? "text-emerald-500"
                        : wsConnected
                        ? "text-amber-500"
                        : "text-slate-300"
                    }`}
                  />
                  {telemetry?.ros_connected
                    ? "ROS 2 Live"
                    : wsConnected
                    ? "Awaiting Gazebo"
                    : "Connecting…"}
                </span>
                <span className="text-slate-300">·</span>
                <span className="inline-flex items-center gap-1 font-mono">
                  <BatteryCharging className="h-3 w-3 text-emerald-600" />
                  {(telemetry?.battery_pct ?? 94).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>

          {/* Center Minimalist Tab Navigation */}
          <nav className="order-3 flex w-full items-center gap-1 overflow-x-auto rounded-xl bg-slate-100/80 p-1 sm:order-2 sm:w-auto">
            {TABS.map((t) => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`inline-flex shrink-0 items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                  tab === t.key
                    ? "bg-white text-slate-900 shadow-2xs"
                    : "text-slate-500 hover:text-slate-800"
                }`}
              >
                {t.icon}
                {t.label}
              </button>
            ))}
          </nav>

          {/* Right Controls: Run Selector + Global E-STOP */}
          <div className="order-2 flex items-center gap-2 sm:order-3">
            <select
              value={selectedRunId ?? ""}
              onChange={(e) => setSelectedRunId(e.target.value ? Number(e.target.value) : null)}
              className="max-w-[190px] truncate rounded-xl border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs font-medium text-slate-700 focus:border-emerald-500 focus:outline-none"
            >
              {runs.length === 0 ? (
                <option value="">No trials recorded yet</option>
              ) : (
                runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    #{r.id} · {r.name}
                  </option>
                ))
              )}
            </select>

            <button
              onClick={() => api.setEstop(!estopActive)}
              className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-semibold transition ${
                estopActive
                  ? "bg-amber-500 text-white hover:bg-amber-600"
                  : "bg-red-50 text-red-700 border border-red-200 hover:bg-red-600 hover:text-white"
              }`}
              title="Emergency Stop"
            >
              <OctagonAlert className="h-3.5 w-3.5" />
              {estopActive ? "STOPPED" : "E-STOP"}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        {tab === "overview" && (
          <OverviewTab
            stats={stats}
            detections={detections}
            telemetry={telemetry}
            onInspectDetection={(det) => {
              setInspectDet(det);
              setTab("detections");
            }}
            onJumpToTab={(t) => setTab(t)}
          />
        )}
        {tab === "heatmap" && (
          <HeatmapTab
            heatmap={heatmap}
            telemetry={telemetry}
            onUpdateDetection={handleUpdateDetection}
          />
        )}
        {tab === "live" && (
          <LiveControlTab
            activeRunId={selectedRunId}
            telemetry={telemetry}
            onRefreshData={() => {
              loadRuns().catch(() => {});
              if (selectedRunId !== null) loadRunData(selectedRunId).catch(() => {});
            }}
          />
        )}
        {tab === "detections" && (
          <DetectionsTab
            detections={detections}
            initialSelected={inspectDet}
            onClearInitialSelected={() => setInspectDet(null)}
            onUpdateDetection={handleUpdateDetection}
          />
        )}
        {tab === "history" && (
          <HistoryLogsTab
            runs={runs}
            selectedRunId={selectedRunId}
            events={events}
            onSelectRun={(id) => setSelectedRunId(id)}
            onCreateRun={handleCreateRun}
            onCompleteRun={handleCompleteRun}
            onDeleteRun={handleDeleteRun}
            onResetDatabase={handleResetDatabase}
          />
        )}
      </main>
    </div>
  );
}
