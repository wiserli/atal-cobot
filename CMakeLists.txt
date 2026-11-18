cmake_minimum_required(VERSION 3.8)
project(wiserli_atal_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    config
    launch
    meshes
    mujoco
    rviz
    urdf
    worlds
  DESTINATION share/${PROJECT_NAME}
)

# Register GZ_SIM_RESOURCE_PATH so Gazebo Harmonic resolves package:// URIs
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/${PROJECT_NAME}.dsv.in")

ament_package()
