#!/usr/bin/env python3
"""
Generate realistic textures, split STL sub-meshes by material (galvanized steel vs white grow-bags),
and build an enhanced, colorful agricultural greenhouse environment in Gazebo Sim.
"""

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

ROW_X_CENTERS = [0.1018, 0.8585, 1.9418, 2.6985]
ROW_X_RANGES = [(0.00, 0.21), (0.75, 0.97), (1.83, 2.05), (2.59, 2.81)]
PANEL_Z_CENTER = 1.0904
PANELS_PER_ROW = 36
Y_START = -0.20
Y_STEP = 1.00


def generate_textures():
    os.makedirs(TEXTURES_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    # 1. Outer Agriculture Field Grass & Soil Texture (1024x1024)
    h, w = 1024, 1024
    base_grass = np.zeros((h, w, 3), dtype=np.float32)
    # BGR in OpenCV: lush natural meadow green with warm soil patches
    base_grass[:, :] = [48, 118, 72]
    noise_fine = rng.normal(0, 14, (h, w, 3)).astype(np.float32)
    noise_coarse = cv2.GaussianBlur(
        rng.normal(0, 35, (h // 8, w // 8, 3)).astype(np.float32), (0, 0), 3
    )
    noise_coarse = cv2.resize(noise_coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    # Add subtle agricultural field plow/furrow lines along Y
    x_grid = np.linspace(0, 32 * np.pi, w, dtype=np.float32)
    furrows = (np.sin(x_grid) * 8.0)[None, :, None]
    field_img = np.clip(base_grass + noise_fine + noise_coarse + furrows, 20, 220).astype(
        np.uint8
    )
    cv2.imwrite(os.path.join(TEXTURES_DIR, "field_grass_soil.png"), field_img)

    # 2. Greenhouse Floor Matting + Soil + Fallen Leaves Texture (1024x1024)
    floor = np.zeros((h, w, 3), dtype=np.float32)
    # Dark earthy brown-gray woven ground cover (BGR)
    floor[:, :] = [58, 66, 72]
    # Woven grid pattern
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
    # Scatter fallen green/brown dried strawberry leaves across the greenhouse floor
    for _ in range(900):
        cx = int(rng.integers(10, w - 10))
        cy = int(rng.integers(10, h - 10))
        ax1 = int(rng.integers(3, 10))
        ax2 = int(rng.integers(2, 6))
        angle = int(rng.integers(0, 180))
        if rng.random() < 0.65:
            # Greenish fallen leaf
            color = (
                int(rng.integers(35, 65)),
                int(rng.integers(85, 135)),
                int(rng.integers(55, 95)),
            )
        else:
            # Dried brown leaf
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
    soil[:, :] = [34, 52, 78]  # Rich dark peat brown (BGR)
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
        # Small white strawberry blossoms & tiny unripe/ripe berries for realism
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
    """Create a 3D rounded/trapezoidal strawberry foliage hedge mesh running along Y."""
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
    """Split 2_farming_rows.stl into white grow-bag troughs vs galvanized steel support structure."""
    src_stl = os.path.join(MESHES_DIR, "2_farming_rows.stl")
    steel_stl = os.path.join(MESHES_DIR, "rows_steel_structure.stl")
    growbag_stl = os.path.join(MESHES_DIR, "rows_white_growbags.stl")

    steel_tris = []
    growbag_tris = []

    with open(src_stl, "rb") as f:
        header = f.read(80)
        num_tri = struct.unpack("<I", f.read(4))[0]
        for _ in range(num_tri):
            raw = f.read(50)
            data = struct.unpack("<12fH", raw)
            x_avg = (data[3] + data[6] + data[9]) / 3.0
            z_avg = (data[5] + data[8] + data[11]) / 3.0
            in_trough_x = any(xmin <= x_avg <= xmax for xmin, xmax in ROW_X_RANGES)
            # Upper trough walls & bags (z_avg >= 0.875 inside the 4 row X ranges)
            if in_trough_x and z_avg >= 0.875:
                growbag_tris.append(raw)
            else:
                steel_tris.append(raw)

    for out_path, tris in [(steel_stl, steel_tris), (growbag_stl, growbag_tris)]:
        with open(out_path, "wb") as f:
            f.write(header)
            f.write(struct.pack("<I", len(tris)))
            for t in tris:
                f.write(t)

    print(
        f"Split STL into steel structure ({len(steel_tris)} tris) and white grow-bags ({len(growbag_tris)} tris)."
    )


def generate_model_sdf():
    stl_full_uri = f"file://{MESHES_DIR}/2_farming_rows.stl"
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

    # 1. Greenhouse Floor Matting under the 4 rows (X centered at 1.40m, Y centered at 17.30m)
    visual_blocks.append(f"""      <!-- Realistic Greenhouse Floor Matting with Fallen Leaves & Soil -->
      <visual name="greenhouse_floor_visual">
        <pose>1.40 17.30 0.002 0 0 0</pose>
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

    # 2. Coco-peat soil strips, lush green foliage hedges, black drip lines, and end catch buckets for each of the 4 rows
    for row_idx, x_pos in enumerate(ROW_X_CENTERS):
        visual_blocks.append(f"""      <!-- Row {row_idx + 1} Coco-Peat Substrate Strip -->
      <visual name="row_{row_idx + 1}_soil_strip">
        <pose>{x_pos:.4f} 17.30 0.991 0 0 0</pose>
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
      <!-- Row {row_idx + 1} Lush Green Strawberry Foliage Canopy -->
      <visual name="row_{row_idx + 1}_foliage_canopy">
        <pose>{x_pos:.4f} 17.30 0.992 0 0 0</pose>
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
      </visual>
      <!-- Row {row_idx + 1} Front & Rear Black Drainage Catch Buckets (like reference photo) -->
      <visual name="row_{row_idx + 1}_front_bucket">
        <pose>{x_pos:.4f} -1.24 0.70 0 0 0</pose>
        <geometry>
          <box>
            <size>0.22 0.24 0.16</size>
          </box>
        </geometry>
        <material>
          <ambient>0.08 0.08 0.09 1</ambient>
          <diffuse>0.12 0.12 0.14 1</diffuse>
          <specular>0.25 0.25 0.25 1</specular>
        </material>
      </visual>
      <visual name="row_{row_idx + 1}_rear_bucket">
        <pose>{x_pos:.4f} 35.84 0.70 0 0 0</pose>
        <geometry>
          <box>
            <size>0.22 0.24 0.16</size>
          </box>
        </geometry>
        <material>
          <ambient>0.08 0.08 0.09 1</ambient>
          <diffuse>0.12 0.12 0.14 1</diffuse>
          <specular>0.25 0.25 0.25 1</specular>
        </material>
      </visual>""")

    # 3. Greenhouse Galvanized Steel Side Posts & Overhead Roof Trusses spanning over the 4 rows
    for arch_idx, y_arch in enumerate(range(-1, 37, 5)):
        visual_blocks.append(f"""      <!-- Greenhouse Steel Arch Frame {arch_idx + 1} at Y={y_arch}m -->
      <visual name="gh_post_left_{arch_idx}">
        <pose>-0.80 {y_arch:.2f} 1.40 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.03</radius>
            <length>2.80</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.55 0.58 0.62 1</ambient>
          <diffuse>0.70 0.73 0.76 1</diffuse>
          <specular>0.6 0.6 0.6 1</specular>
        </material>
      </visual>
      <visual name="gh_post_right_{arch_idx}">
        <pose>3.60 {y_arch:.2f} 1.40 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.03</radius>
            <length>2.80</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.55 0.58 0.62 1</ambient>
          <diffuse>0.70 0.73 0.76 1</diffuse>
          <specular>0.6 0.6 0.6 1</specular>
        </material>
      </visual>
      <visual name="gh_truss_beam_{arch_idx}">
        <pose>1.40 {y_arch:.2f} 2.80 0 1.5707963 0</pose>
        <geometry>
          <cylinder>
            <radius>0.025</radius>
            <length>4.44</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.55 0.58 0.62 1</ambient>
          <diffuse>0.70 0.73 0.76 1</diffuse>
          <specular>0.6 0.6 0.6 1</specular>
        </material>
      </visual>""")

    # 4. 144 Strawberry 20x20cm (640x640) Inspection Panels equally distributed across the 4 rows
    for row_idx, x_pos in enumerate(ROW_X_CENTERS):
        row_cls_seq = []
        for _ in range(PANELS_PER_ROW // 4):
            block = list(cls_keys)
            rng.shuffle(block)
            row_cls_seq.extend(block)

        for panel_idx, cls_key in enumerate(row_cls_seq):
            y_pos = Y_START + panel_idx * Y_STEP
            img_list = classes[cls_key]
            img_name = img_list[cls_counters[cls_key] % len(img_list)]
            cls_counters[cls_key] += 1
            img_uri = f"file://{TEST_IMAGES_DIR}/{img_name}"

            visual_name = f"strawberry_r{row_idx + 1}_p{panel_idx + 1:02d}_{cls_key}"
            visual_blocks.append(f"""      <visual name="{visual_name}">
        <pose>{x_pos:.4f} {y_pos:.4f} {PANEL_Z_CENTER:.4f} 0 0 0</pose>
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
            <uri>{stl_full_uri}</uri>
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
    write_tiled_obj(os.path.join(MESHES_DIR, "greenhouse_floor.obj"), 5.2, 40.0, 4.0, 30.0)
    write_tiled_obj(os.path.join(MESHES_DIR, "soil_strip.obj"), 0.17, 36.8, 1.0, 180.0)
    write_canopy_hedge_obj(os.path.join(MESHES_DIR, "foliage_hedge.obj"), 0.22, 36.6, 0.085, 90.0)
    split_stl_by_material()
    generate_model_sdf()


if __name__ == "__main__":
    main()
