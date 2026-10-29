#!/home/burneye228/ros2_ws/.venv/bin/python
"""
1. Generate top-down PGM occupancy grid map & costmap of the 4 strawberry rows from 2_farming_rows_updated.STL.
2. Generate 24 ArUco marker textures (DICT_4X4_50, IDs 0..23) and landmark config placed every 5m across Left, Center, and Right lanes.
"""

import os
import struct
import numpy as np
import cv2

PACKAGE_DIR = "/home/burneye228/ros2_ws/src/agriculture_sim"
MESHES_DIR = os.path.join(PACKAGE_DIR, "models/farming_rows/meshes")
ARUCO_TEX_DIR = os.path.join(MESHES_DIR, "textures/aruco")
MAPS_DIR = os.path.join(PACKAGE_DIR, "maps")
CONFIG_DIR = os.path.join(PACKAGE_DIR, "config")
STL_PATH = os.path.join(MESHES_DIR, "2_farming_rows_updated.STL")

# Map bounds & resolution (2 cm / pixel)
MAP_RES = 0.02
X_MIN, X_MAX = -2.00, 4.80
Y_MIN, Y_MAX = -4.00, 36.50

# 3 Empty Drivable Lanes (Left of Row 1, Center between Row 2 & 3, Right of Row 4)
LANE_X = {
    "left": -0.4400,
    "center": 1.4000,
    "right": 3.2400,
}

# Every 5.0 meters along the 35m rows (from Y = -1.0m to Y = 34.0m)
ARUCO_Y_POSITIONS = [-1.0, 4.0, 9.0, 14.0, 19.0, 24.0, 29.0, 34.0]
ARUCO_SIZE_M = 0.18  # 18cm x 18cm marker on ground


def generate_pgm_costmap():
    os.makedirs(MAPS_DIR, exist_ok=True)

    width_px = int(round((X_MAX - X_MIN) / MAP_RES))
    height_px = int(round((Y_MAX - Y_MIN) / MAP_RES))

    # Occupancy grid: 254 = free (white), 0 = occupied obstacle (black)
    occ_grid = np.full((height_px, width_px), 254, dtype=np.uint8)

    with open(STL_PATH, "rb") as f:
        f.read(80)
        num_tri = struct.unpack("<I", f.read(4))[0]
        for _ in range(num_tri):
            data = struct.unpack("<12fH", f.read(50))
            pts_px = []
            for i in range(3, 12, 3):
                x, y = data[i], data[i + 1]
                px = int((x - X_MIN) / MAP_RES)
                # In ROS map images, row 0 is at Y_MAX (top-down north-up)
                py = int((Y_MAX - y) / MAP_RES)
                pts_px.append([px, py])
            pts_arr = np.array([pts_px], dtype=np.int32)
            cv2.fillPoly(occ_grid, pts_arr, 0)

    # Save raw binary occupancy grid PGM (for nav2_map_server)
    pgm_path = os.path.join(MAPS_DIR, "farming_rows_map.pgm")
    cv2.imwrite(pgm_path, occ_grid)

    # Compute Euclidean distance transform for Costmap representation (inflation radius = 0.35m)
    binary_free = (occ_grid > 128).astype(np.uint8)
    dist_px = cv2.distanceTransform(binary_free, cv2.DIST_L2, 5)
    dist_m = dist_px * MAP_RES

    costmap_pgm = np.full_like(occ_grid, 254)
    # Lethal obstacle = 0, Inscribed radius (0.22m) = 30, Exponential decay out to 0.40m
    inscribed_r = 0.22
    inflation_r = 0.42
    costmap_pgm[dist_m <= 0.01] = 0
    inscribed_mask = (dist_m > 0.01) & (dist_m <= inscribed_r)
    costmap_pgm[inscribed_mask] = 40
    decay_mask = (dist_m > inscribed_r) & (dist_m <= inflation_r)
    norm_d = (dist_m[decay_mask] - inscribed_r) / (inflation_r - inscribed_r)
    costmap_pgm[decay_mask] = (40 + norm_d * 214).astype(np.uint8)

    costmap_pgm_path = os.path.join(MAPS_DIR, "farming_rows_costmap.pgm")
    cv2.imwrite(costmap_pgm_path, costmap_pgm)

    # Write ROS 2 map_server YAML files
    yaml_content = f"""image: farming_rows_map.pgm
mode: trinary
resolution: {MAP_RES}
origin: [{X_MIN}, {Y_MIN}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    with open(os.path.join(MAPS_DIR, "farming_rows_map.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_content)

    costmap_yaml_content = f"""image: farming_rows_costmap.pgm
mode: scale
resolution: {MAP_RES}
origin: [{X_MIN}, {Y_MIN}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    with open(os.path.join(MAPS_DIR, "farming_rows_costmap.yaml"), "w", encoding="utf-8") as f:
        f.write(costmap_yaml_content)

    # Also save a colorized visualization showing rows + costmap inflation + ArUco positions
    color_vis = cv2.cvtColor(costmap_pgm, cv2.COLOR_GRAY2BGR)
    color_vis[decay_mask] = [180, 210, 255]     # Light orange/amber inflation zone
    color_vis[inscribed_mask] = [80, 120, 255]  # Inscribed safety buffer
    color_vis[dist_m <= 0.01] = [20, 20, 30]    # Black obstacles (4 rows + supports)

    for lane_name, lx in LANE_X.items():
        for y_pos in ARUCO_Y_POSITIONS:
            px = int((lx - X_MIN) / MAP_RES)
            py = int((Y_MAX - y_pos) / MAP_RES)
            cv2.rectangle(color_vis, (px - 5, py - 5), (px + 5, py + 5), (0, 200, 0), -1)

    cv2.imwrite(os.path.join(MAPS_DIR, "farming_rows_costmap_preview.png"), color_vis)
    print(f"Saved PGM maps ({width_px}x{height_px} px at {MAP_RES} m/px) in {MAPS_DIR}")


def generate_aruco_markers_and_config():
    os.makedirs(ARUCO_TEX_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Create flat horizontal 18cm x 18cm OBJ mesh for ArUco markers
    half = ARUCO_SIZE_M / 2.0
    aruco_obj_path = os.path.join(MESHES_DIR, "aruco_marker.obj")
    obj_content = f"""# Horizontal 18x18cm ArUco Marker on Ground (+Z normal)
v -{half:.4f} -{half:.4f} 0.0
v  {half:.4f} -{half:.4f} 0.0
v  {half:.4f}  {half:.4f} 0.0
v -{half:.4f}  {half:.4f} 0.0

vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0

vn 0.0 0.0 1.0

f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
"""
    with open(aruco_obj_path, "w", encoding="utf-8") as f:
        f.write(obj_content)

    landmarks = []
    marker_id = 0

    for lane_name in ["left", "center", "right"]:
        lx = LANE_X[lane_name]
        for y_pos in ARUCO_Y_POSITIONS:
            # Generate 512x512 marker with crisp white quiet-zone border (70px margin)
            canvas = np.full((512, 512, 3), 255, dtype=np.uint8)
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 372)
            canvas[70:442, 70:442] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
            # Outer thin dark frame for visual crispness on ground
            cv2.rectangle(canvas, (4, 4), (507, 507), (40, 40, 40), 4)

            tex_name = f"aruco_id{marker_id:02d}.png"
            tex_path = os.path.join(ARUCO_TEX_DIR, tex_name)
            cv2.imwrite(tex_path, canvas)

            landmarks.append(
                {
                    "id": marker_id,
                    "lane": lane_name,
                    "x": lx,
                    "y": float(y_pos),
                    "z": 0.004,
                    "yaw": 1.5707963,
                    "texture": tex_path,
                }
            )
            marker_id += 1

    # Write EKF / Navigation landmark configuration YAML
    yaml_lines = [
        "# Ground-truth ArUco landmark coordinates (DICT_4X4_50) placed every 5.0m across all 3 lanes",
        "aruco_dictionary: DICT_4X4_50",
        f"marker_size_m: {ARUCO_SIZE_M}",
        "lanes:",
        "  left: -0.4400",
        "  center: 1.4000",
        "  right: 3.2400",
        "markers:",
    ]
    for lm in landmarks:
        yaml_lines.append(f"  - id: {lm['id']}")
        yaml_lines.append(f"    lane: {lm['lane']}")
        yaml_lines.append(f"    pose: [{lm['x']:.4f}, {lm['y']:.4f}, {lm['z']:.4f}, 0.0, 0.0, {lm['yaw']:.7f}]")

    landmarks_yaml_path = os.path.join(CONFIG_DIR, "aruco_landmarks.yaml")
    with open(landmarks_yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(f"Generated {len(landmarks)} ArUco marker textures and {landmarks_yaml_path}")
    return landmarks


if __name__ == "__main__":
    generate_pgm_costmap()
    generate_aruco_markers_and_config()
