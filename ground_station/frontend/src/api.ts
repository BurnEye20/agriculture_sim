import type {
  Detection,
  GstStatusResponse,
  HeatmapPayload,
  MissionEvent,
  Run,
  RunStats,
} from "./types";

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listRuns: () => req<{ runs: Run[] }>("/api/runs"),
  getEmptyHeatmap: () => req<HeatmapPayload>("/api/field/empty_heatmap"),
  resetDatabase: () => req<{ reset: boolean }>("/api/database/reset", { method: "POST" }),
  createRun: (name: string, notes: string) =>
    req<Run>("/api/runs", { method: "POST", body: JSON.stringify({ name, notes }) }),
  completeRun: (runId: number, status = "completed") =>
    req<Run>(`/api/runs/${runId}/complete`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    }),
  deleteRun: (runId: number) =>
    req<{ deleted: boolean }>(`/api/runs/${runId}`, { method: "DELETE" }),

  getRunStats: (runId: number) => req<RunStats>(`/api/runs/${runId}/stats`),
  getRunHeatmap: (runId: number) => req<HeatmapPayload>(`/api/runs/${runId}/heatmap`),
  getRunDetections: (runId: number, params?: { only_diseased?: boolean }) => {
    const qs = params?.only_diseased ? "?only_diseased=true" : "";
    return req<{ detections: Detection[] }>(`/api/runs/${runId}/detections${qs}`);
  },
  updateDetection: (detId: number, body: { reviewed?: boolean; operator_note?: string }) =>
    req<Detection>(`/api/detections/${detId}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  getRunLogs: (runId: number, category = "all") =>
    req<{ events: MissionEvent[] }>(
      `/api/runs/${runId}/logs?category=${encodeURIComponent(category)}`
    ),

  getGstStatus: () => req<GstStatusResponse>("/api/gstreamer/status"),
  setGstPipeline: (channel: string, pipeline: string) =>
    req<{ ok: boolean; message: string }>("/api/gstreamer/pipeline", {
      method: "POST",
      body: JSON.stringify({ channel, pipeline }),
    }),

  sendCmdVel: (linear_x: number, angular_z: number, duration_s = 0.55) =>
    req<{ ok: boolean; ros_connected?: boolean; reason?: string }>("/api/control/cmd_vel", {
      method: "POST",
      body: JSON.stringify({ linear_x, angular_z, duration_s }),
    }),
  setEstop: (active: boolean) =>
    req<{ ok: boolean; ros_connected?: boolean; estop_active: boolean }>("/api/control/estop", {
      method: "POST",
      body: JSON.stringify({ active }),
    }),
  setMastHeight: (height_m: number, auto_align?: boolean) =>
    req<{
      ok: boolean;
      ros_connected?: boolean;
      mast_target_m: number;
      auto_align_mast: boolean;
      reason?: string;
    }>("/api/control/mast_height", {
      method: "POST",
      body: JSON.stringify({ height_m, auto_align }),
    }),
  sendMission: (action: string, run_id?: number) =>
    req<{
      ok: boolean;
      ros_connected?: boolean;
      message: string;
      active_run_id?: number | null;
      new_run_created?: boolean;
    }>("/api/control/mission", {
      method: "POST",
      body: JSON.stringify({ action, run_id }),
    }),
  captureSnapshot: (
    run_id: number | null,
    camera: string,
    disease_class: string,
    confidence = 0.9
  ) =>
    req<Detection>("/api/control/snapshot", {
      method: "POST",
      body: JSON.stringify({ run_id, camera, disease_class, confidence }),
    }),
};

export function imageUrl(imagePath: string): string {
  if (!imagePath) return "";
  return `/api/images/${imagePath}`;
}

export function streamUrl(camera: "left" | "right" | "bottom"): string {
  return `/api/stream/${camera}`;
}

export function telemetryWsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/telemetry`;
}
