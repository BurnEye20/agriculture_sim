import React, { useState } from "react";
import {
  CheckCircle2,
  Database,
  Download,
  Plus,
  Terminal,
  Trash2,
} from "lucide-react";
import type { MissionEvent, Run } from "../types";

interface Props {
  runs: Run[];
  selectedRunId: number | null;
  events: MissionEvent[];
  onSelectRun: (runId: number) => void;
  onCreateRun: (name: string, notes: string) => Promise<void>;
  onCompleteRun: (runId: number) => Promise<void>;
  onDeleteRun: (runId: number) => Promise<void>;
  onResetDatabase: () => Promise<void>;
}

export const HistoryLogsTab: React.FC<Props> = ({
  runs,
  selectedRunId,
  events,
  onSelectRun,
  onCreateRun,
  onCompleteRun,
  onDeleteRun,
  onResetDatabase,
}) => {
  const [newName, setNewName] = useState("");
  const [newNotes, setNewNotes] = useState("");
  const [catFilter, setCatFilter] = useState("all");
  const [creating, setCreating] = useState(false);

  const filteredEvents = events.filter(
    (e) => catFilter === "all" || e.category === catFilter
  );

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    try {
      await onCreateRun(newName.trim(), newNotes.trim());
      setNewName("");
      setNewNotes("");
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Start New Inspection Run (persisted in SQLite) */}
      <form
        onSubmit={handleCreate}
        className="flex flex-col gap-3 rounded-2xl border border-slate-200/90 bg-white p-4 sm:flex-row sm:items-end"
      >
        <div className="flex-1">
          <label className="mb-1 block text-xs font-medium text-slate-600">
            New Inspection Run Name
          </label>
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="e.g., Afternoon Sweep · Block A #04"
            className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-900 focus:border-emerald-500 focus:bg-white focus:outline-none"
          />
        </div>
        <div className="flex-1">
          <label className="mb-1 block text-xs font-medium text-slate-600">
            Operator Notes
          </label>
          <input
            value={newNotes}
            onChange={(e) => setNewNotes(e.target.value)}
            placeholder="Weather, irrigation state, or test goal…"
            className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-900 focus:border-emerald-500 focus:bg-white focus:outline-none"
          />
        </div>
        <button
          type="submit"
          disabled={creating || !newName.trim()}
          className="inline-flex items-center justify-center gap-1.5 rounded-xl bg-slate-900 px-4 py-2 text-xs font-semibold text-white transition hover:bg-slate-800 disabled:opacity-40"
        >
          <Plus className="h-3.5 w-3.5" /> Start & Log Run
        </button>
      </form>

      {/* Runs Database Table */}
      <div className="overflow-hidden rounded-2xl border border-slate-200/90 bg-white">
        <div className="flex items-center justify-between border-b border-slate-100 px-5 py-3.5">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-emerald-600" />
            <h3 className="text-sm font-semibold text-slate-900">
              Archived & Active Runs (SQLite)
            </h3>
          </div>
          <div className="flex items-center gap-3">
            <span className="font-mono text-xs text-slate-400">
              {runs.length} recorded run{runs.length === 1 ? "" : "s"}
            </span>
            {runs.length > 0 && (
              <button
                onClick={onResetDatabase}
                className="rounded-lg border border-red-200 bg-red-50 px-2.5 py-1 text-[11px] font-medium text-red-700 hover:bg-red-100"
              >
                Clear All Data
              </button>
            )}
          </div>
        </div>

        <div className="overflow-x-auto">
          {runs.length === 0 ? (
            <div className="p-10 text-center text-xs text-slate-500">
              No runs in SQLite yet (0 runs · 0 captures). Start a run above or launch Gazebo
              to auto-record your first live trial.
            </div>
          ) : (
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-100 bg-slate-50/70 text-[11px] uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="px-5 py-3 font-medium">Run</th>
                  <th className="px-4 py-3 font-medium">Status</th>
                  <th className="px-4 py-3 font-medium">Health</th>
                  <th className="px-4 py-3 font-medium">Diseased / Total</th>
                  <th className="px-4 py-3 font-medium">Distance</th>
                  <th className="px-4 py-3 font-medium">Avg Mast</th>
                  <th className="px-5 py-3 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {runs.map((r) => {
                  const isSelected = r.id === selectedRunId;
                  const healthPct =
                    r.total_detections > 0
                      ? `${Math.round((100 * r.healthy_count) / r.total_detections)}%`
                      : "—";
                  return (
                    <tr
                      key={r.id}
                      onClick={() => onSelectRun(r.id)}
                      className={`cursor-pointer transition ${
                        isSelected ? "bg-emerald-50/40" : "hover:bg-slate-50/70"
                      }`}
                    >
                      <td className="px-5 py-3.5">
                        <div className="font-semibold text-slate-900">{r.name}</div>
                        <div className="mt-0.5 text-[11px] text-slate-400">
                          {new Date(r.started_at).toLocaleString()}
                          {r.notes ? ` · ${r.notes}` : ""}
                        </div>
                      </td>
                      <td className="px-4 py-3.5">
                        <span
                          className={`inline-flex rounded-full px-2 py-0.5 text-[11px] font-medium capitalize ${
                            r.status === "active"
                              ? "bg-emerald-100 text-emerald-800"
                              : "bg-slate-100 text-slate-600"
                          }`}
                        >
                          {r.status}
                        </span>
                      </td>
                      <td className="px-4 py-3.5 font-mono font-semibold text-emerald-700">
                        {healthPct}
                      </td>
                      <td className="px-4 py-3.5 font-mono text-slate-700">
                        <span className="font-semibold text-red-600">{r.diseased_count}</span> /{" "}
                        {r.total_detections}
                      </td>
                      <td className="px-4 py-3.5 font-mono text-slate-600">
                        {r.distance_m.toFixed(1)} m
                      </td>
                      <td className="px-4 py-3.5 font-mono text-slate-600">
                        {r.avg_mast_height_m > 0 ? `${r.avg_mast_height_m.toFixed(3)} m` : "—"}
                      </td>
                      <td
                        className="px-5 py-3.5 text-right"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <div className="inline-flex items-center gap-1.5">
                          {r.status === "active" && (
                            <button
                              onClick={() => onCompleteRun(r.id)}
                              title="Complete Run"
                              className="rounded-lg border border-emerald-200 bg-emerald-50 p-1.5 text-emerald-700 hover:bg-emerald-100"
                            >
                              <CheckCircle2 className="h-3.5 w-3.5" />
                            </button>
                          )}
                          <a
                            href={`/api/runs/${r.id}/export.csv`}
                            download
                            title="Export CSV"
                            className="rounded-lg border border-slate-200 bg-white p-1.5 text-slate-600 hover:bg-slate-50"
                          >
                            <Download className="h-3.5 w-3.5" />
                          </a>
                          <button
                            onClick={() => onDeleteRun(r.id)}
                            title="Delete Run"
                            className="rounded-lg border border-slate-200 bg-white p-1.5 text-slate-400 hover:border-red-200 hover:text-red-600"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Mission & System Event Log */}
      <div className="rounded-2xl border border-slate-200/90 bg-white p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Terminal className="h-4 w-4 text-slate-500" />
            <h3 className="text-sm font-semibold text-slate-900">
              Run Event & Telemetry Log
            </h3>
          </div>
          <div className="flex flex-wrap gap-1">
            {["all", "mission", "detection", "mast", "nav", "gstreamer"].map((c) => (
              <button
                key={c}
                onClick={() => setCatFilter(c)}
                className={`rounded-md px-2 py-1 text-[11px] font-medium uppercase ${
                  catFilter === c
                    ? "bg-slate-900 text-white"
                    : "text-slate-500 hover:bg-slate-100"
                }`}
              >
                {c}
              </button>
            ))}
          </div>
        </div>

        <div className="max-h-72 space-y-1.5 overflow-y-auto rounded-xl border border-slate-100 bg-slate-50/60 p-3 font-mono text-xs">
          {filteredEvents.length === 0 ? (
            <div className="py-6 text-center text-slate-400">No events logged.</div>
          ) : (
            filteredEvents.map((ev) => (
              <div
                key={ev.id}
                className="flex flex-wrap items-baseline gap-2 rounded-lg bg-white px-3 py-1.5 border border-slate-100"
              >
                <span className="text-[11px] text-slate-400">
                  {new Date(ev.timestamp).toLocaleTimeString()}
                </span>
                <span
                  className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${
                    ev.level === "alert"
                      ? "bg-red-50 text-red-700"
                      : ev.level === "warn"
                      ? "bg-amber-50 text-amber-700"
                      : "bg-slate-100 text-slate-600"
                  }`}
                >
                  {ev.category}
                </span>
                <span className="flex-1 text-slate-700">{ev.message}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
