export type TabKey = "overview" | "heatmap" | "live" | "detections" | "history";

export interface Run {
  id: number;
  name: string;
  started_at: string;
  ended_at: string | null;
  status: "active" | "completed" | "aborted" | string;
  field_name: string;
  slope_deg: number;
  distance_m: number;
  duration_s: number;
  total_detections: number;
  diseased_count: number;
  healthy_count: number;
  avg_confidence: number;
  avg_mast_height_m: number;
  notes: string;
}

export interface Detection {
  id: number;
  run_id: number;
  timestamp: string;
  disease_class: "anthracnose_fruit_rot" | "gray_mold" | "powdery_mildew" | "healthy" | string;
  severity: "high" | "medium" | "low" | "none" | string;
  confidence: number;
  x_m: number;
  y_m: number;
  row_index: number;
  lane: string;
  camera_side: string;
  mast_height_m: number;
  image_path: string;
  bbox: number[][];
  reviewed: boolean;
  operator_note: string;
}

export interface DiseaseBreakdownItem {
  key: string;
  name: string;
  full_name: string;
  count: number;
  color: string;
  pct: number;
}

export interface RowBreakdownItem {
  row: string;
  row_index: number;
  healthy: number;
  anthracnose_fruit_rot: number;
  gray_mold: number;
  powdery_mildew: number;
  diseased_total: number;
  total: number;
}

export interface ZoneProfileItem {
  zone: string;
  y_start: number;
  diseased: number;
  healthy: number;
}

export interface RunStats {
  run: Run;
  health_score_pct: number | null;
  total_detections: number;
  diseased_count: number;
  healthy_count: number;
  reviewed_diseased: number;
  disease_breakdown: DiseaseBreakdownItem[];
  row_breakdown: RowBreakdownItem[];
  zone_profile: ZoneProfileItem[];
  odom_drift: {
    max_drift_m: number;
    avg_drift_m: number;
    samples: number;
  };
}

export interface FieldGeometry {
  name: string;
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
  row_y_min: number;
  row_y_max: number;
  rows: { index: number; x: number; slope_deg: number; label: string }[];
  lanes: { name: string; x: number; label: string }[];
  aruco_y: number[];
  mast_min_m: number;
  mast_max_m: number;
}

export interface ArucoLandmark {
  id: number;
  lane: string;
  x: number;
  y: number;
}

export interface HeatmapPayload {
  field: FieldGeometry;
  aruco_landmarks: ArucoLandmark[];
  detections: Detection[];
  ekf_path: { x: number; y: number }[];
  raw_odom_path: { x: number; y: number }[];
}

export interface MissionEvent {
  id: number;
  run_id: number;
  timestamp: string;
  level: "info" | "warn" | "alert" | string;
  category: "mission" | "detection" | "mast" | "nav" | "gstreamer" | string;
  message: string;
}

export interface GstChannelStatus {
  channel: string;
  mode: string;
  ros_live: boolean;
  fps: number;
  resolution: string;
  codec: string;
  gst_version: string;
  pipeline: string;
  error: string | null;
}

export interface GstStatusResponse {
  gst_available: boolean;
  gst_version: string;
  channels: Record<"left" | "right" | "bottom", GstChannelStatus>;
}

export interface Telemetry {
  ros_connected: boolean;
  ros_available?: boolean;
  mission_state: string;
  robot_mode?: string;
  status_detail?: string;
  estop_active: boolean;
  x_m: number;
  y_m: number;
  yaw_rad: number;
  yaw_deg: number;
  linear_vel: number;
  angular_vel: number;
  raw_odom_x: number;
  raw_odom_y: number;
  drift_m: number;
  mast_height_m: number;
  mast_target_m: number;
  auto_align_mast: boolean;
  last_aruco_id: number | null;
  aruco_dist_m: number;
  battery_pct: number;
  slope_deg: number;
  active_run_id?: number | null;
  timestamp: number;
  gst?: Record<string, { fps: number; mode: string; ros_live: boolean }>;
}

export const DISEASE_STYLES: Record<
  string,
  { label: string; short: string; color: string; bg: string; text: string; border: string }
> = {
  anthracnose_fruit_rot: {
    label: "Anthracnose Fruit Rot",
    short: "Anthracnose",
    color: "#dc2626",
    bg: "bg-red-50",
    text: "text-red-700",
    border: "border-red-200",
  },
  gray_mold: {
    label: "Gray Mold (Botrytis)",
    short: "Gray Mold",
    color: "#d97706",
    bg: "bg-amber-50",
    text: "text-amber-700",
    border: "border-amber-200",
  },
  powdery_mildew: {
    label: "Powdery Mildew",
    short: "Powdery Mildew",
    color: "#7c3aed",
    bg: "bg-violet-50",
    text: "text-violet-700",
    border: "border-violet-200",
  },
  healthy: {
    label: "Healthy Fruit",
    short: "Healthy",
    color: "#059669",
    bg: "bg-emerald-50",
    text: "text-emerald-700",
    border: "border-emerald-200",
  },
};
