cmake_minimum_required(VERSION 3.10)
project(mujoco_rl_training)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(mujoco_core REQUIRED)
find_package(mujoco_viewer REQUIRED)

if(DEFINED ENV{MUJOCO_PATH} AND NOT "$ENV{MUJOCO_PATH}" STREQUAL "")
  set(MUJOCO_PATH "$ENV{MUJOCO_PATH}")
elseif(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
  set(MUJOCO_PATH "$ENV{CONDA_PREFIX}")
endif()

if(NOT EXISTS "${MUJOCO_PATH}/include/mujoco/mujoco.h")
  message(FATAL_ERROR
    "MuJoCo headers not found at ${MUJOCO_PATH}/include/mujoco/mujoco.h. "
    "Set MUJOCO_PATH to your MuJoCo install or build inside pixi.")
endif()

add_library(mujoco_rl_training SHARED
  src/PendulumEnv.cpp
)

target_include_directories(mujoco_rl_training PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${MUJOCO_PATH}/include/mujoco>
  $<BUILD_INTERFACE:${MUJOCO_PATH}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(mujoco_rl_training
  mujoco_core::mujoco_core
)

ament_target_dependencies(mujoco_rl_training
  mujoco_core
)

add_executable(pendulum_rl_demo
  src/pendulum_rl_demo.cpp
)

add_executable(pendulum_rl_visual_demo
  src/pendulum_rl_visual_demo.cpp
)

target_include_directories(pendulum_rl_demo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${MUJOCO_PATH}/include/mujoco>
  $<BUILD_INTERFACE:${MUJOCO_PATH}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(pendulum_rl_demo
  mujoco_rl_training
  mujoco_core::mujoco_core
)

ament_target_dependencies(pendulum_rl_demo
  ament_index_cpp
  mujoco_core
)

target_include_directories(pendulum_rl_visual_demo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${MUJOCO_PATH}/include/mujoco>
  $<BUILD_INTERFACE:${MUJOCO_PATH}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(pendulum_rl_visual_demo
  mujoco_rl_training
  mujoco_viewer::mujoco_viewer
  mujoco_core::mujoco_core
)

ament_target_dependencies(pendulum_rl_visual_demo
  ament_index_cpp
  mujoco_core
  mujoco_viewer
)

install(
  TARGETS mujoco_rl_training
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS pendulum_rl_demo
          pendulum_rl_visual_demo
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
