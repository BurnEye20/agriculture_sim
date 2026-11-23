# AgriRover — Autonomous Greenhouse Strawberry Inspection & Ground Station (`agriculture_sim`)

A complete **ROS 2 Jazzy + Gazebo Sim + Nav2 + YOLOv11 + Web Ground Station** simulation and operator control suite for an autonomous 4-wheel agricultural inspection rover operating in an uneven hydroponic strawberry greenhouse.

---

## Key Capabilities

### 1. Realistic Uneven Greenhouse Environment (Gazebo Sim)
- **Sloped Agricultural Terrain**: Entire greenhouse field tilted at a **$2.0^\circ$ overall steepness** ($\text{roll}=1.732^\circ$, $\text{pitch}=-1.000^\circ$) with **65 non-uniform physical soil bumps & ridges** across the 3 driving lanes (`Left`, `Center`, `Right`).
- **Uneven Hydroponic Strawberry Aisles**: 4 elevated $35.0\text{ m}$ rows (`Row 1`–`Row 4`) inclined at **$\pm 0.25^\circ$** with per-row height offsets (`~0.86 m – 1.05 m` strawberry height variation along the aisles), galvanized steel support stands, white hydroponic grow-bags, lush foliage canopy hedges, and **144 inspection panels** drawn from a 4-class strawberry disease dataset.
- **24 Ground ArUco Landmarks**: `DICT_4X4_50` markers spaced every $5.0\text{ m}$ across all 3 lanes (`config/aruco_landmarks.yaml`).

### 2. Standard ROS 2 Navigation & Realistic Localization Stack
- **Nav2 Autonomous Multi-Row Inspection**: Full Nav2 stack (`map_server`, `costmap_server`, `planner_server`, `controller_server` with `RegulatedPurePursuitController`, `behavior_server`, `bt_navigator`, `waypoint_follower`, `lifecycle_manager`) driven by `scripts/nav2_row_mission.py` across all 3 lanes and back to base (`(-0.44, -1.75 m)`), with live operator pause, RTB, reset, and E-STOP control over `/ground_station/mission_cmd`.
- **Sensor Fusion (`robot_localization` EKF)**: `ekf_node` (`world_frame: map`, `config/ekf.yaml`) fusing wheel velocities (`/rover/wheel_odom`), gyroscope yaw rate (`/rover/imu_noisy`), and perspective-rectified ArUco pose corrections (`/aruco_pose`) computed from the upper-frame tilted ArUco camera (`48.7°` downward pitch) via 4-corner ray-plane homography + `cv2.solvePnP` (`IPPE_SQUARE`) in `scripts/aruco_localizer_node.py`.
- **Realistic Odometry & IMU Drift Model**: `scripts/wheel_odom_noise_node.py` injects realistic **$2.5\%\text{–}4.5\%$ wheel-slip longitudinal drift** (amplified live when IMU vertical acceleration / roll-pitch jolts detect soil bumps), cross-slope creep, and MEMS gyroscope drift + Gaussian noise, while publishing `/ekf_path`, `/odom_imu_encoder_path`, and `/ground_truth_path` for RViz & Ground Station drift comparison.

### 3. 3-Section Telescopic Camera Mast & Visual-Servoing Auto-Alignment
- **Parametric Linear-Actuator Lifting Column**: Generated from `config/camera_mast.yaml` via `scripts/generate_camera_mast.py` (3 telescopic frames, $1.20\text{ m}$ physical max height, **$0.726\text{ m} \to 1.105\text{ m}$** camera optical height range) and driven synchronously by `scripts/camera_mast_controller.py` on `/camera_mast/set_height`.
- **Current-Row Visual Servoing & YOLOv11 Disease Detection**: `scripts/yolo_strawberry_detector.py` runs `YOLOv11` (`best.pt`: *Anthracnose Fruit Rot*, *Gray Mold*, *Powdery Mildew*, *Healthy Fruit*) on both left and right row cameras (equipped with a **$0.98\text{ m}$ optical depth gate** to isolate only the adjacent current row and exclude rows 2–4 and outer grass). It features:
  - Cloud-immune green foliage canopy & foreground detection visual servoing to automatically adjust mast height along the uneven $\pm 0.25^\circ$ aisles (with manual override via `/camera_mast/auto_align`).
  - Blank-sky fast-path skip + `depth=1` QoS + confidence-adaptive confirmation (instant confirm at $\ge 0.58$, 2-of-3 frame voting for $0.45\text{–}0.58$) and synchronized JPEG capture publishing on `/rover/yolo_detections`.

### 4. Minimalistic Ground Station Web Application (`ground_station/`)
Built with **React 19 + TypeScript + Tailwind CSS + Recharts** and a **FastAPI + SQLite + native GStreamer 1.24 + ROS 2** backend:
- **Overview Tab**: Live KPIs (Field Health Index, Disease Detections, Coverage & EKF vs. Raw Odometry Drift, Telescopic Mast Height), Disease Breakdown donut chart, Row-by-Row disease distribution, Longitudinal 0–35 m aisle profile, and clickable recent captures.
- **Field Heatmap Tab**: Interactive $35.0\text{ m} \times 5.2\text{ m}$ 2D greenhouse map displaying all 4 uneven strawberry rows, 3 lanes, 24 ArUco landmarks, toggleable Gaussian disease hotspots, **EKF vs. Drifting Raw Odometry** trajectories, live rover pose, and a side-panel camera shot & coordinate inspector.
- **Live & Control Tab**:
  - **GStreamer 1.24 Live Video Streaming** (`appsrc ! videoconvert ! jpegenc ! appsink`) for Left, Right, Dual Side-by-Side Row, and Bottom Tilted ArUco cameras, plus an expandable **custom GStreamer pipeline configurator** (`udpsrc`/RTP VP8, `rtspsrc`, `v4l2src`, `videotestsrc`) for physical rover deployment.
  - **Live Robot Status Card** (*Running Mission*, *Returning to Base*, *Manual Mode*, *At Base · Ready*, *E-STOP*), Mission Start/Pause/RTB buttons, Emergency Stop, WASD + pointer-capture teleop pad (10 Hz dead-man watchdog), and **Telescopic Camera Mast** slider + Auto-Align toggle.
- **Captures Tab**: Filterable disease photo gallery (by disease class, row 1–4, or search) showing synchronized YOLOv11 bounding-box captures, exact `(X, Y)` coordinates, mast height, and SQLite-persisted **Reviewed** status & operator treatment notes.
- **Runs & Logs Tab**: Starts cleanly at zero data; automatically records live Gazebo trials into SQLite (`runs`, `detections`, `telemetry_logs`, `mission_events`), automatically archives a run upon returning to base (auto-RTB or manual) so the next **Start** creates `Run #2`, `#3`, …, and supports CSV export and full DB reset.

---

## Repository Structure

```text
agriculture_sim/
├── CMakeLists.txt & package.xml
├── config/
│   ├── agriculture_bridge.yaml       # ros_gz_bridge topic mappings
│   ├── aruco_landmarks.yaml          # 24 ground ArUco marker coordinates
│   ├── camera_mast.yaml              # Parametric 3-section telescopic mast config
│   ├── camera_mast_derived.yaml      # Auto-generated mast kinematics
│   ├── ekf.yaml                      # robot_localization EKF config (world_frame: map)
│   └── nav2_params.yaml              # Nav2 planner, RPP controller, costmaps & behaviors
├── launch/
│   └── agriculture_world.launch.xml  # Launches Gazebo, Bridge, EKF, Nav2, Mast, YOLO & RViz2
├── maps/                             # Static occupancy map & costmap for Nav2 map_server
├── models/
│   ├── agri_rover/                   # 4WD rover SDF + telescopic mast + 3 cameras + IMU
│   └── farming_rows/                 # 4 uneven strawberry rows, 65 soil bumps, 24 ArUco codes
├── rviz/
│   └── agriculture_config.rviz       # Pre-configured RViz2 layout (paths, map, 3 cameras)
├── scripts/
│   ├── aruco_localizer_node.py       # Tilted-camera ArUco 3D pose localizer (/aruco_pose)
│   ├── build_realistic_environment.py# Procedural textures, 0.25 deg warped STL rows & SDF builder
│   ├── camera_mast_controller.py     # Linear-actuator telescopic mast height controller
│   ├── generate_camera_mast.py       # Generates mast SDF/URDF from camera_mast.yaml
│   ├── nav2_row_mission.py           # Nav2 multi-row mission node + Ground Station command listener
│   ├── wheel_odom_noise_node.py      # 2.5%-4.5% bump/slope wheel & IMU drift simulator
│   └── yolo_strawberry_detector.py   # YOLOv11 disease detector + visual-servoing mast auto-align
├── urdf/
│   └── agri_rover.urdf               # Robot description for robot_state_publisher / RViz
├── worlds/
│   └── agriculture_world.sdf         # 2.0 deg sloped greenhouse world
└── ground_station/                   # React + TypeScript + FastAPI + GStreamer + SQLite Web App
    ├── start_ground_station.sh       # One-command backend + frontend launcher
    ├── backend/
    │   ├── main.py                   # FastAPI REST, WebSocket (/ws/telemetry) & MJPEG server
    │   ├── database.py               # SQLite schema, CRUD, analytics & CSV export
    │   ├── gstreamer_streamer.py     # Native GStreamer 1.24 pipeline manager
    │   ├── ros_bridge.py             # Bidirectional ROS 2 bridge & live SQLite trial recorder
    │   └── requirements.txt
    └── frontend/                     # Vite + React 19 + TypeScript + Tailwind CSS + Recharts
        ├── package.json
        └── src/
```

---

## Prerequisites

- **OS & ROS 2**: Ubuntu 24.04 with **ROS 2 Jazzy** and **Gazebo Harmonic** (`ros-jazzy-ros-gz`, `ros-jazzy-navigation2`, `ros-jazzy-nav2-bringup`, `ros-jazzy-robot-localization`, `ros-jazzy-cv-bridge`)
- **GStreamer**: `gstreamer1.0-tools`, `gstreamer1.0-plugins-base`, `gstreamer1.0-plugins-good`, `python3-gi`, `gir1.2-gstreamer-1.0`
- **Python Environment** (`~/ros2_ws/.venv` or system Python 3.12): `ultralytics`, `opencv-python`, `numpy`, `pyyaml`, `fastapi`, `uvicorn[standard]`, `python-multipart`
- **Node.js**: Node.js `v20+` / `v22 LTS` & `npm` (for building/running `ground_station/frontend`)

---

## Quick Start

### 1. Build the ROS 2 Simulation Package
```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select agriculture_sim
source install/setup.bash
```

### 2. Launch the Greenhouse Simulation + Nav2 + YOLOv11 + RViz2
```bash
ros2 launch agriculture_sim agriculture_world.launch.xml
```

### 3. Launch the Ground Station Web Application
In a second terminal (in the same `ROS_DOMAIN_ID`):
```bash
# Install backend dependencies once (if not already in your venv)
pip install -r ~/ros2_ws/src/agriculture_sim/ground_station/backend/requirements.txt

# Start FastAPI + GStreamer 1.24 + SQLite backend (:8000) & React Vite UI (:5173)
~/ros2_ws/src/agriculture_sim/ground_station/start_ground_station.sh
```
Open **`http://localhost:5173`** (or build production static assets with `cd ground_station/frontend && npm run build` and access directly on **`http://localhost:8000`**).

---

## Key ROS 2 Topics

| Topic | Type | Description |
| :--- | :--- | :--- |
| `/odometry/filtered` | `nav_msgs/Odometry` | `robot_localization` EKF pose in `map` frame |
| `/rover/wheel_odom` | `nav_msgs/Odometry` | Noisy wheel odometry (`2.5%–4.5%` bump/slope drift) |
| `/rover/imu_noisy` | `sensor_msgs/Imu` | Corrupted IMU with bump jolts & MEMS gyro drift |
| `/aruco_pose` | `geometry_msgs/PoseWithCovarianceStamped` | Tilted-camera ArUco absolute pose corrections |
| `/cmd_vel` | `geometry_msgs/Twist` | Rover velocity commands (Nav2 or Ground Station teleop) |
| `/camera_mast/set_height` | `std_msgs/Float64` | Desired camera optical height (`0.726 m – 1.105 m`) |
| `/camera_mast/current_height` | `std_msgs/Float64` | Measured telescopic mast camera height |
| `/camera_mast/auto_align` | `std_msgs/Bool` | Enables/disables visual-servoing mast auto-alignment |
| `/ground_station/mission_cmd` | `std_msgs/String` | `start`, `pause`, `rtb`, `reset`, `stop`, `estop`, `release_estop` |
| `/ground_station/mission_status` | `std_msgs/String` | Live operating mode & active waypoint detail |
| `/rover/yolo_detections` | `std_msgs/String` | Confirmed YOLOv11 JSON detections + synchronized capture JPEG |
| `/rover/{left,right}_camera/detections/image_raw` | `sensor_msgs/Image` | Annotated row camera feeds (ingested by GStreamer) |
| `/rover/bottom_camera/aruco_detections/image_raw` | `sensor_msgs/Image` | Annotated tilted ArUco camera feed |

---

## Regenerating Parametric Models (Optional)
- **Modify telescopic mast geometry** (`config/camera_mast.yaml`):
  ```bash
  python3 scripts/generate_camera_mast.py
  cd ~/ros2_ws && colcon build --packages-select agriculture_sim
  ```
- **Modify uneven aisle angles / textures** (`scripts/build_realistic_environment.py`):
  ```bash
  python3 scripts/build_realistic_environment.py
  cd ~/ros2_ws && colcon build --packages-select agriculture_sim
  ```
