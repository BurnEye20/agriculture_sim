cmake_minimum_required(VERSION 3.8)
project(agriculture_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  launch
  models
  rviz
  urdf
  worlds
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/yolo_strawberry_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in")

ament_package()
