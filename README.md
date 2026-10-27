# agriculture_sim

Gazebo Sim models and worlds for the vertical strawberry farming agriculture project in ROS 2.

## Contents

- `models/farming_rows/` — Gazebo model (`model.sdf`, `model.config`, and `meshes/2_farming_rows.stl`) representing 4 rows of vertical strawberry systems.
- `worlds/agriculture_world.sdf` — Gazebo Sim world containing the ground plane, lighting, physics, sensors, and the `farming_rows` model.
- `launch/agriculture_world.launch.xml` — ROS 2 launch file to start Gazebo Sim with `agriculture_world.sdf`.

## Usage

### Build
```bash
cd ~/ros2_ws
colcon build --packages-select agriculture_sim
source install/setup.bash
```

### Launch via ROS 2
```bash
ros2 launch agriculture_sim agriculture_world.launch.xml
```
