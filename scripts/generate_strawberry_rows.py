#!/usr/bin/env python3
"""Generate farming_rows/model.sdf with 20x20 cm (640x640) strawberry texture panels on top of the 4 rows."""

import os
import random

PACKAGE_DIR = "/home/burneye228/ros2_ws/src/agriculture_sim"
TEST_IMAGES_DIR = os.path.join(PACKAGE_DIR, "Final_Fruit_Dataset_YOLOv11/test/images")
MODEL_SDF_PATH = os.path.join(PACKAGE_DIR, "models/farming_rows/model.sdf")
STL_URI = f"file://{PACKAGE_DIR}/models/farming_rows/meshes/2_farming_rows.stl"
PANEL_OBJ_URI = f"file://{PACKAGE_DIR}/models/farming_rows/meshes/strawberry_panel.obj"

# Center X coordinates of the 4 strawberry rows in 2_farming_rows.stl
ROW_X_CENTERS = [0.1018, 0.8585, 1.9418, 2.6985]

# Top of the rows is at Z = 0.9904 m; panel height is 0.20 m (-0.10 to +0.10 m),
# so placing center at Z = 1.0904 m puts the bottom of the 20 cm image flush on top of the row.
PANEL_Z_CENTER = 1.0904

# 36 panels per row (9 of each of the 4 classes per row), spaced 1.0 m apart along Y
PANELS_PER_ROW = 36
Y_START = -0.20
Y_STEP = 1.00


def main():
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

    for row_idx, x_pos in enumerate(ROW_X_CENTERS):
        # Build a balanced sequence of 36 classes for this row (9 blocks of all 4 classes)
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
            <uri>{PANEL_OBJ_URI}</uri>
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
      <visual name="rows_visual">
        <geometry>
          <mesh>
            <uri>{STL_URI}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>0.75 0.78 0.75 1</ambient>
          <diffuse>0.82 0.85 0.82 1</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
        </material>
      </visual>
      <collision name="rows_collision">
        <geometry>
          <mesh>
            <uri>{STL_URI}</uri>
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

    print(f"Generated {MODEL_SDF_PATH} with {len(visual_blocks)} strawberry panels.")
    print("Panels per class across all 4 rows:", cls_counters)


if __name__ == "__main__":
    main()
