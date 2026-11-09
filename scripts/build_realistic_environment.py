#!/usr/bin/env python3
"""
Generate realistic textures, split 2_farming_rows_updated.STL into sub-meshes by material
(galvanized steel vs white grow-bags), and build the agricultural greenhouse environment in Gazebo Sim.
"""

import math
import os
import random
import struct
import numpy as np
import cv2

PACKAGE_DIR = "/home/burneye228/ros2_ws/src/agriculture_sim"
MESHES_DIR = os.path.join(PACKAGE_DIR, "models/farming_rows/meshes")
TEXTURES_DIR = os.path.join(MESHES_DIR, "textures")
TEST_IMAGES_DIR = os.path.join(PACKAGE_DIR, "Final_Fruit_Dataset_YOLOv11/test/images")
MODEL_SDF_PATH = os.path.join(PACKAGE_DIR, "models/farming_rows/model.sdf")

UPDATED_STL_NAME = "2_farming_rows_updated.STL"

# Refined 35.0 m row dimensions from 2_farming_rows_updated.STL:
# X in [0.0045, 2.7958], Y in [-1.3925, 33.6075] (length = 35.0 m, midpoint Y = 16.1075 m), Z in [0.0, 0.9904]
ROW_X_CENTERS = [0.1018, 0.8585, 1.9418, 2.6985]
ROW_X_RANGES = [(0.00, 0.21), (0.75, 0.97), (1.83, 2.05), (2.59, 2.81)]
ROW_Y_MIN = -1.3925
ROW_Y_MAX = 33.6075
ROW_Y_MID = (ROW_Y_MIN + ROW_Y_MAX) / 2.0  # 16.1075 m
ROW_LENGTH = ROW_Y_MAX - ROW_Y_MIN         # 35.0 m

ORIG_TROUGH_Z = 0.9904
PANEL_OFFSET_ABOVE_TROUGH = 0.1000
PANELS_PER_ROW = 36
Y_START = -0.60
Y_STEP = 0.955

# Uneven strawberry aisle configuration (smooth 0.25 degree aisle inclination + subtle per-row offsets)
# Over the 35 m row length, 0.25 deg produces a natural ~15.3 cm elevation change (panels spanning
# ~0.86 m to ~1.06 m), sitting smoothly between the previous extremes and within the telescopic
# camera mast's 0.726 m - 1.105 m optical height adjustment range.
AISLE_ANGLE_DEG = 0.25
AISLE_ANGLE_RAD = math.radians(AISLE_ANGLE_DEG)
AISLE_SLOPE_TAN = math.tan(AISLE_ANGLE_RAD)
BASE_TROUGH_Z_MID = 0.8550
ROW_SLOPE_SIGNS = [1.0, 1.0, -1.0, -1.0]         # Left pair rises +0.25 deg, right pair descends -0.25 deg
ROW_Z_OFFSETS = [-0.020, 0.015, -0.010, 0.025]   # Smooth uneven base heights across the 4 rows (m)


def row_trough_z(row_idx: int, y: float) -> float:
    """Return the top-of-trough Z height (m) for row `row_idx` (0..3) at coordinate `y`."""
    return (
        BASE_TROUGH_Z_MID
        + ROW_Z_OFFSETS[row_idx]
        + ROW_SLOPE_SIGNS[row_idx] * (y - ROW_Y_MID) * AISLE_SLOPE_TAN
    )


def row_panel_z(row_idx: int, y: float) -> float:
    """Return the strawberry inspection panel center Z height (m) for row `row_idx` at `y`."""
    return row_trough_z(row_idx, y) + PANEL_OFFSET_ABOVE_TROUGH


def row_roll_rad(row_idx: int) -> float:
    """Return the SDF roll angle (rad around +X) that tilts a Y-aligned row by +/-1.0 deg."""
    return ROW_SLOPE_SIGNS[row_idx] * AISLE_ANGLE_RAD


def target_trough_z_at_xy(x: float, y: float) -> float:
    """Continuous target trough top Z(x, y) across the left (rows 0,1) and right (rows 2,3) stands."""
    if x < 1.40:
        w = min(max((x - 0.25) / (0.71 - 0.25), 0.0), 1.0)
        return (1.0 - w) * row_trough_z(0, y) + w * row_trough_z(1, y)
    else:
        w = min(max((x - 2.09) / (2.55 - 2.09), 0.0), 1.0)
        return (1.0 - w) * row_trough_z(2, y) + w * row_trough_z(3, y)


def warp_stl_vertex_z(x: float, y: float, z: float) -> float:
    """
    Transform vertex Z in 2_farming_rows_updated.STL to follow the 1.0 deg uneven aisle profile:
    - Base plates at z == 0 remain flush on the ground (z_new = 0).
    - Upper gutter channels & grow-bags (z >= 0.85 m) shift rigidly by delta_z(x, y).
    - Support legs & diagonal braces (0 < z < 0.85 m) scale linearly from 0 to 0.85 + delta_z(x, y).
    """
    if z <= 1e-5:
        return 0.0
    delta_z = target_trough_z_at_xy(x, y) - ORIG_TROUGH_Z
    z_join = 0.85
    if z >= z_join:
        return z + delta_z
    return z * ((z_join + delta_z) / z_join)


def generate_textures():
    os.makedirs(TEXTURES_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    # 1. Outer Agriculture Field Grass & Soil Texture (1024x1024)
    h, w = 1024, 1024
    base_grass = np.zeros((h, w, 3), dtype=np.float32)
    base_grass[:, :] = [48, 118, 72]
    noise_fine = rng.normal(0, 14, (h, w, 3)).astype(np.float32)
    noise_coarse = cv2.GaussianBlur(
        rng.normal(0, 35, (h // 8, w // 8, 3)).astype(np.float32), (0, 0), 3
    )
    noise_coarse = cv2.resize(noise_coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    x_grid = np.linspace(0, 32 * np.pi, w, dtype=np.float32)
    furrows = (np.sin(x_grid) * 8.0)[None, :, None]
    field_img = np.clip(base_grass + noise_fine + noise_coarse + furrows, 20, 220).astype(
        np.uint8
    )
    cv2.imwrite(os.path.join(TEXTURES_DIR, "field_grass_soil.png"), field_img)

    # 2. Greenhouse Floor Matting + Soil + Fallen Leaves Texture (1024x1024)
    floor = np.zeros((h, w, 3), dtype=np.float32)
    floor[:, :] = [58, 66, 72]
    for i in range(0, h, 8):
        floor[i : i + 2, :] += 6
    for j in range(0, w, 8):
        floor[:, j : j + 2] += 6
    soil_patches = cv2.GaussianBlur(
        rng.normal(0, 28, (h // 4, w // 4, 3)).astype(np.float32), (0, 0), 4
    )
    soil_patches = cv2.resize(soil_patches, (w, h), interpolation=cv2.INTER_CUBIC)
    floor = np.clip(floor + soil_patches + rng.normal(0, 7, (h, w, 3)), 30, 150).astype(
        np.uint8
    )
    for _ in range(900):
        cx = int(rng.integers(10, w - 10))
        cy = int(rng.integers(10, h - 10))
        ax1 = int(rng.integers(3, 10))
        ax2 = int(rng.integers(2, 6))
        angle = int(rng.integers(0, 180))
        if rng.random() < 0.65:
            color = (
                int(rng.integers(35, 65)),
                int(rng.integers(85, 135)),
                int(rng.integers(55, 95)),
            )
        else:
            color = (
                int(rng.integers(30, 50)),
                int(rng.integers(65, 95)),
                int(rng.integers(95, 135)),
            )
        cv2.ellipse(floor, (cx, cy), (ax1, ax2), angle, 0, 360, color, -1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(TEXTURES_DIR, "greenhouse_floor.png"), floor)

    # 3. Rich Coco-Peat / Organic Substrate Texture (512x512)
    sh, sw = 512, 512
    soil = np.zeros((sh, sw, 3), dtype=np.float32)
    soil[:, :] = [34, 52, 78]
    soil += rng.normal(0, 14, (sh, sw, 3))
    soil = np.clip(soil, 15, 130).astype(np.uint8)
    cv2.imwrite(os.path.join(TEXTURES_DIR, "coco_peat_soil.png"), soil)

    # 4. Lush Green Strawberry Foliage Canopy Texture (512x512)
    foliage = np.zeros((sh, sw, 3), dtype=np.uint8)
    foliage[:, :] = (38, 105, 52)
    for _ in range(1800):
        cx = int(rng.integers(0, sw))
        cy = int(rng.integers(0, sh))
        r1 = int(rng.integers(6, 20))
        r2 = int(rng.integers(4, 14))
        ang = int(rng.integers(0, 180))
        g_val = int(rng.integers(95, 175))
        b_val = int(rng.integers(25, 65))
        r_val = int(rng.integers(35, 95))
        cv2.ellipse(
            foliage, (cx, cy), (r1, r2), ang, 0, 360, (b_val, g_val, r_val), -1, cv2.LINE_AA
        )
        if rng.random() < 0.03:
            cv2.circle(foliage, (cx, cy), 4, (220, 245, 250), -1, cv2.LINE_AA)
            cv2.circle(foliage, (cx, cy), 2, (40, 210, 245), -1, cv2.LINE_AA)
        elif rng.random() < 0.025:
            cv2.circle(foliage, (cx, cy), 5, (35, 35, 210), -1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(TEXTURES_DIR, "strawberry_foliage.png"), foliage)

    print("Generated procedural agricultural textures in:", TEXTURES_DIR)


def write_tiled_obj(path, width_x, length_y, uv_scale_x, uv_scale_y):
    hx = width_x / 2.0
    hy = length_y / 2.0
    content = f"""# Textured Ground Plane OBJ ({width_x}m x {length_y}m)
v -{hx:.4f} -{hy:.4f} 0.0
v  {hx:.4f} -{hy:.4f} 0.0
v  {hx:.4f}  {hy:.4f} 0.0
v -{hx:.4f}  {hy:.4f} 0.0

vt 0.0 0.0
vt {uv_scale_x:.2f} 0.0
vt {uv_scale_x:.2f} {uv_scale_y:.2f}
vt 0.0 {uv_scale_y:.2f}

vn 0.0 0.0 1.0

f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_canopy_hedge_obj(path, width_x, length_y, height_z, uv_y):
    hx_bot = width_x / 2.0
    hx_top = width_x * 0.38
    hy = length_y / 2.0
    content = f"""# 3D Strawberry Foliage Canopy Hedge along Y
v -{hx_bot:.4f} -{hy:.4f} 0.0
v  {hx_bot:.4f} -{hy:.4f} 0.0
v  {hx_top:.4f} -{hy:.4f} {height_z:.4f}
v -{hx_top:.4f} -{hy:.4f} {height_z:.4f}
v -{hx_bot:.4f}  {hy:.4f} 0.0
v  {hx_bot:.4f}  {hy:.4f} 0.0
v  {hx_top:.4f}  {hy:.4f} {height_z:.4f}
v -{hx_top:.4f}  {hy:.4f} {height_z:.4f}

vt 0.0 0.0
vt {uv_y:.1f} 0.0
vt {uv_y:.1f} 1.0
vt 0.0 1.0

vn -0.85 0.0 0.52
vn  0.85 0.0 0.52
vn  0.0  0.0 1.0

# Left side slope (-X)
f 5/1/1 1/2/1 4/3/1
f 5/1/1 4/3/1 8/4/1

# Right side slope (+X)
f 2/1/2 6/2/2 7/3/2
f 2/1/2 7/3/2 3/4/2

# Top canopy face (+Z)
f 4/1/3 3/2/3 7/3/3
f 4/1/3 7/3/3 8/4/3
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def split_stl_by_material():
    """
    Split 2_farming_rows_updated.STL into white grow-bag troughs vs galvanized steel support
    structure while warping the 4 rows to the 1.0 deg uneven aisle height profile (keeping
    all steel support feet at z = 0 on the sloped ground).
    """
    src_stl = os.path.join(MESHES_DIR, UPDATED_STL_NAME)
    steel_stl = os.path.join(MESHES_DIR, "rows_steel_structure.stl")
    growbag_stl = os.path.join(MESHES_DIR, "rows_white_growbags.stl")
    collision_stl = os.path.join(MESHES_DIR, "rows_sloped_collision.stl")

    steel_tris = []
    growbag_tris = []
    all_sloped_tris = []

    with open(src_stl, "rb") as f:
        header = f.read(80)
        num_tri = struct.unpack("<I", f.read(4))[0]
        for _ in range(num_tri):
            raw = f.read(50)
            data = struct.unpack("<12fH", raw)
            x1, y1, z1 = data[3], data[4], data[5]
            x2, y2, z2 = data[6], data[7], data[8]
            x3, y3, z3 = data[9], data[10], data[11]
            attr = data[12]

            x_avg = (x1 + x2 + x3) / 3.0
            z_avg = (z1 + z2 + z3) / 3.0
            in_trough_x = any(xmin <= x_avg <= xmax for xmin, xmax in ROW_X_RANGES)

            nz1 = warp_stl_vertex_z(x1, y1, z1)
            nz2 = warp_stl_vertex_z(x2, y2, z2)
            nz3 = warp_stl_vertex_z(x3, y3, z3)

            # Recompute triangle normal after Z warp
            ux, uy, uz = x2 - x1, y2 - y1, nz2 - nz1
            vx, vy, vz = x3 - x1, y3 - y1, nz3 - nz1
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            norm = math.sqrt(nx * nx + ny * ny + nz * nz)
            if norm > 1e-12:
                nx, ny, nz = nx / norm, ny / norm, nz / norm
            else:
                nx, ny, nz = data[0], data[1], data[2]

            warped_raw = struct.pack(
                "<12fH", nx, ny, nz, x1, y1, nz1, x2, y2, nz2, x3, y3, nz3, attr
            )
            all_sloped_tris.append(warped_raw)
            if in_trough_x and z_avg >= 0.875:
                growbag_tris.append(warped_raw)
            else:
                steel_tris.append(warped_raw)

    for out_path, tris in [
        (steel_stl, steel_tris),
        (growbag_stl, growbag_tris),
        (collision_stl, all_sloped_tris),
    ]:
        with open(out_path, "wb") as f:
            f.write(header)
            f.write(struct.pack("<I", len(tris)))
            for t in tris:
                f.write(t)

    print(
        f"Warped & split {UPDATED_STL_NAME} (1.0 deg uneven aisles): "
        f"steel ({len(steel_tris)} tris), grow-bags ({len(growbag_tris)} tris), "
        f"collision ({len(all_sloped_tris)} tris)."
    )


def generate_model_sdf():
    stl_collision_uri = f"file://{MESHES_DIR}/rows_sloped_collision.stl"
    steel_uri = f"file://{MESHES_DIR}/rows_steel_structure.stl"
    growbag_uri = f"file://{MESHES_DIR}/rows_white_growbags.stl"
    panel_obj_uri = f"file://{MESHES_DIR}/strawberry_panel.obj"
    foliage_obj_uri = f"file://{MESHES_DIR}/foliage_hedge.obj"
    soil_strip_obj_uri = f"file://{MESHES_DIR}/soil_strip.obj"
    gh_floor_obj_uri = f"file://{MESHES_DIR}/greenhouse_floor.obj"

    foliage_tex_uri = f"file://{TEXTURES_DIR}/strawberry_foliage.png"
    soil_tex_uri = f"file://{TEXTURES_DIR}/coco_peat_soil.png"
    gh_floor_tex_uri = f"file://{TEXTURES_DIR}/greenhouse_floor.png"

    all_images = sorted(os.listdir(TEST_IMAGES_DIR))
    classes = {
        "cls0_anthracnose": [f for f in all_images if f.startswith("cls0_")],
        "cls1_gray_mold": [f for f in all_images if f.startswith("cls1_")],
        "cls2_powdery_mildew": [f for f in all_images if f.startswith("cls2_")],
        "cls3_healthy": [f for f in all_images if f.startswith("cls3_")],
    }
    cls_keys = list(classes.keys())
    rng = random.Random(42)
    cls_counters = {k: 0 for k in cls_keys}

    visual_blocks = []

    # 1. Greenhouse Floor Matting under the 4 rows (X centered at 1.40m, Y centered at ROW_Y_MID = 16.1075m)
    visual_blocks.append(f"""      <!-- Realistic Greenhouse Floor Matting with Fallen Leaves & Soil -->
      <visual name="greenhouse_floor_visual">
        <pose>1.40 {ROW_Y_MID:.4f} 0.002 0 0 0</pose>
        <cast_shadows>false</cast_shadows>
        <geometry>
          <mesh>
            <uri>{gh_floor_obj_uri}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>0.85 0.85 0.85 1</ambient>
          <diffuse>0.95 0.95 0.95 1</diffuse>
          <specular>0.08 0.08 0.08 1</specular>
          <pbr>
            <metal>
              <albedo_map>{gh_floor_tex_uri}</albedo_map>
              <roughness>0.92</roughness>
              <metalness>0.0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>""")

    # 2. Coco-peat soil strips and lush green foliage hedges for each of the 4 rows
    #    Each row is inclined at 0.25 deg (roll around +X) with smooth uneven heights across the 4 aisles.
    for row_idx, x_pos in enumerate(ROW_X_CENTERS):
        roll_rad = row_roll_rad(row_idx)
        z_trough_mid = row_trough_z(row_idx, ROW_Y_MID)
        z_soil_mid = z_trough_mid + 0.0006
        z_foliage_mid = z_trough_mid + 0.0016
        slope_deg_signed = ROW_SLOPE_SIGNS[row_idx] * AISLE_ANGLE_DEG

        visual_blocks.append(f"""      <!-- Row {row_idx + 1} Coco-Peat Substrate Strip (Aisle Slope: {slope_deg_signed:+.2f} deg) -->
      <visual name="row_{row_idx + 1}_soil_strip">
        <pose>{x_pos:.4f} {ROW_Y_MID:.4f} {z_soil_mid:.4f} {roll_rad:.7f} 0 0</pose>
        <cast_shadows>false</cast_shadows>
        <geometry>
          <mesh>
            <uri>{soil_strip_obj_uri}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.9 0.9 0.9 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
          <pbr>
            <metal>
              <albedo_map>{soil_tex_uri}</albedo_map>
              <roughness>0.95</roughness>
              <metalness>0.0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>
      <!-- Row {row_idx + 1} Lush Green Strawberry Foliage Canopy (Aisle Slope: {slope_deg_signed:+.2f} deg) -->
      <visual name="row_{row_idx + 1}_foliage_canopy">
        <pose>{x_pos:.4f} {ROW_Y_MID:.4f} {z_foliage_mid:.4f} {roll_rad:.7f} 0 0</pose>
        <cast_shadows>true</cast_shadows>
        <geometry>
          <mesh>
            <uri>{foliage_obj_uri}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>0.75 0.85 0.75 1</ambient>
          <diffuse>0.92 0.98 0.90 1</diffuse>
          <specular>0.12 0.15 0.12 1</specular>
          <pbr>
            <metal>
              <albedo_map>{foliage_tex_uri}</albedo_map>
              <roughness>0.80</roughness>
              <metalness>0.0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>""")

    # 3. 24 Ground ArUco Markers (every 5.0m across Left, Center, and Right lanes)
    aruco_obj_uri = f"file://{MESHES_DIR}/aruco_marker.obj"
    lane_x_map = [("left", -0.4400), ("center", 1.4000), ("right", 3.2400)]
    aruco_y_positions = [-1.0, 4.0, 9.0, 14.0, 19.0, 24.0, 29.0, 34.0]
    marker_id = 0
    for lane_name, lx in lane_x_map:
        for y_pos in aruco_y_positions:
            aruco_tex_uri = f"file://{TEXTURES_DIR}/aruco/aruco_id{marker_id:02d}.png"
            visual_blocks.append(f"""      <!-- Ground ArUco Marker ID {marker_id:02d} ({lane_name} lane, Y={y_pos}m) -->
      <visual name="aruco_marker_id{marker_id:02d}_{lane_name}">
        <pose>{lx:.4f} {y_pos:.4f} 0.004 0 0 0</pose>
        <cast_shadows>false</cast_shadows>
        <geometry>
          <mesh>
            <uri>{aruco_obj_uri}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>1.0 1.0 1.0 1</ambient>
          <diffuse>1.0 1.0 1.0 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
          <pbr>
            <metal>
              <albedo_map>{aruco_tex_uri}</albedo_map>
              <roughness>0.85</roughness>
              <metalness>0.0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>""")
            marker_id += 1

    # 4. 65 Non-Uniform Physical Field Bumps & Soil Ridges (Collision + Visual) across Left, Center, Right lanes
    bump_rng = random.Random(123)
    bump_collisions = []
    bump_idx = 0
    for lane_name, lx in lane_x_map:
        y_cursor = -0.30
        while y_cursor < 33.20:
            y_cursor += bump_rng.uniform(0.65, 1.35)
            # Keep 0.35m clearance around ArUco markers so markers sit cleanly on the sloped ground
            if any(abs(y_cursor - ay) < 0.38 for ay in aruco_y_positions):
                continue

            # Place bump asymmetrically under left wheel (-0.25m), right wheel (+0.25m), or across full lane (0.0m)
            track_choice = bump_rng.choice(["left_wheel", "right_wheel", "full_lane"])
            if track_choice == "left_wheel":
                offset_x = -0.25
                bump_len = bump_rng.uniform(0.32, 0.42)
            elif track_choice == "right_wheel":
                offset_x = 0.25
                bump_len = bump_rng.uniform(0.32, 0.42)
            else:
                offset_x = 0.0
                bump_len = bump_rng.uniform(0.62, 0.72)

            bx = lx + offset_x
            by = y_cursor
            # Smooth soil ridge: height 0.9 cm to 1.8 cm above ground (radius 0.22m to 0.32m buried cylinder)
            bump_r = bump_rng.uniform(0.22, 0.32)
            bump_protrusion = bump_rng.uniform(0.009, 0.018)
            bz = -bump_r + bump_protrusion
            bump_yaw = bump_rng.uniform(-0.18, 0.18)

            bump_idx += 1
            visual_blocks.append(f"""      <!-- Field Soil Bump {bump_idx} ({lane_name} lane, Y={by:.2f}m) -->
      <visual name="field_bump_vis_{bump_idx}">
        <pose>{bx:.4f} {by:.4f} {bz:.4f} 0 1.5707963 {bump_yaw:.4f}</pose>
        <geometry>
          <cylinder>
            <radius>{bump_r:.4f}</radius>
            <length>{bump_len:.4f}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.32 0.25 0.18 1</ambient>
          <diffuse>0.42 0.33 0.24 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
          <pbr>
            <metal>
              <albedo_map>{soil_tex_uri}</albedo_map>
              <roughness>0.95</roughness>
              <metalness>0.0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>""")
            bump_collisions.append(f"""      <collision name="field_bump_col_{bump_idx}">
        <pose>{bx:.4f} {by:.4f} {bz:.4f} 0 1.5707963 {bump_yaw:.4f}</pose>
        <geometry>
          <cylinder>
            <radius>{bump_r:.4f}</radius>
            <length>{bump_len:.4f}</length>
          </cylinder>
        </geometry>
      </collision>""")

    visual_blocks.extend(bump_collisions)

    # 5. 144 Strawberry 20x20cm (640x640) Inspection Panels equally distributed across the 4 uneven 1.0 deg rows
    for row_idx, x_pos in enumerate(ROW_X_CENTERS):
        roll_rad = row_roll_rad(row_idx)
        row_cls_seq = []
        for _ in range(PANELS_PER_ROW // 4):
            block = list(cls_keys)
            rng.shuffle(block)
            row_cls_seq.extend(block)

        for panel_idx, cls_key in enumerate(row_cls_seq):
            y_pos = Y_START + panel_idx * Y_STEP
            z_panel = row_panel_z(row_idx, y_pos)
            img_list = classes[cls_key]
            img_name = img_list[cls_counters[cls_key] % len(img_list)]
            cls_counters[cls_key] += 1
            img_uri = f"file://{TEST_IMAGES_DIR}/{img_name}"

            visual_name = f"strawberry_r{row_idx + 1}_p{panel_idx + 1:02d}_{cls_key}"
            visual_blocks.append(f"""      <visual name="{visual_name}">
        <pose>{x_pos:.4f} {y_pos:.4f} {z_panel:.4f} {roll_rad:.7f} 0 0</pose>
        <cast_shadows>false</cast_shadows>
        <geometry>
          <mesh>
            <uri>{panel_obj_uri}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>1.0 1.0 1.0 1</ambient>
          <diffuse>1.0 1.0 1.0 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
          <pbr>
            <metal>
              <albedo_map>{img_uri}</albedo_map>
              <roughness>0.9</roughness>
              <metalness>0.0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>""")

    visuals_str = "\n".join(visual_blocks)

    sdf_content = f"""<?xml version="1.0" ?>
<sdf version="1.10">
  <model name="farming_rows">
    <static>true</static>
    <pose>0 0 0 0 0 0</pose>
    <link name="rows_link">
      <!-- Galvanized Steel Support Stands, Legs & Lower Gutter Channels -->
      <visual name="rows_steel_structure_visual">
        <geometry>
          <mesh>
            <uri>{steel_uri}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>0.52 0.56 0.60 1</ambient>
          <diffuse>0.68 0.72 0.76 1</diffuse>
          <specular>0.65 0.65 0.65 1</specular>
          <pbr>
            <metal>
              <roughness>0.35</roughness>
              <metalness>0.75</metalness>
            </metal>
          </pbr>
        </material>
      </visual>

      <!-- White Hydroponic Strawberry Grow-Bags -->
      <visual name="rows_white_growbags_visual">
        <geometry>
          <mesh>
            <uri>{growbag_uri}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>0.90 0.90 0.88 1</ambient>
          <diffuse>0.97 0.97 0.95 1</diffuse>
          <specular>0.25 0.25 0.25 1</specular>
          <pbr>
            <metal>
              <roughness>0.65</roughness>
              <metalness>0.05</metalness>
            </metal>
          </pbr>
        </material>
      </visual>

      <collision name="rows_collision">
        <geometry>
          <mesh>
            <uri>{stl_collision_uri}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </collision>

{visuals_str}
    </link>
  </model>
</sdf>
"""
    with open(MODEL_SDF_PATH, "w", encoding="utf-8") as f:
        f.write(sdf_content)
    print("Updated", MODEL_SDF_PATH)


def main():
    generate_textures()
    write_tiled_obj(os.path.join(MESHES_DIR, "field_ground.obj"), 120.0, 120.0, 30.0, 30.0)
    write_tiled_obj(os.path.join(MESHES_DIR, "greenhouse_floor.obj"), 5.2, 38.0, 4.0, 28.0)
    write_tiled_obj(os.path.join(MESHES_DIR, "soil_strip.obj"), 0.17, 34.8, 1.0, 170.0)
    write_canopy_hedge_obj(os.path.join(MESHES_DIR, "foliage_hedge.obj"), 0.22, 34.6, 0.085, 85.0)
    split_stl_by_material()
    generate_model_sdf()


if __name__ == "__main__":
    main()
