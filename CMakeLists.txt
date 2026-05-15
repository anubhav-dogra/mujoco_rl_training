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
find_package(Torch REQUIRED)

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

function(target_mujoco_include_dirs target)
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/envs/include>
    $<BUILD_INTERFACE:${MUJOCO_PATH}/include/mujoco>
    $<BUILD_INTERFACE:${MUJOCO_PATH}/include>
    $<INSTALL_INTERFACE:include>
  )
endfunction()

function(add_rl_executable target source)
  add_executable(${target} ${source})
  target_mujoco_include_dirs(${target})
  target_link_libraries(${target}
    mujoco_rl_training
    mujoco_core::mujoco_core
  )
  ament_target_dependencies(${target}
    ament_index_cpp
    mujoco_core
  )
  list(APPEND RL_EXECUTABLE_TARGETS ${target})
  set(RL_EXECUTABLE_TARGETS "${RL_EXECUTABLE_TARGETS}" PARENT_SCOPE)
endfunction()

function(add_rl_viewer_executable target source)
  add_executable(${target} ${source})
  target_mujoco_include_dirs(${target})
  target_link_libraries(${target}
    mujoco_rl_training
    mujoco_viewer::mujoco_viewer
    mujoco_core::mujoco_core
  )
  ament_target_dependencies(${target}
    ament_index_cpp
    mujoco_core
    mujoco_viewer
  )
  list(APPEND RL_EXECUTABLE_TARGETS ${target})
  set(RL_EXECUTABLE_TARGETS "${RL_EXECUTABLE_TARGETS}" PARENT_SCOPE)
endfunction()

# building torch examples with dependencies and linking with Torch libraries
function(add_torch_example target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE
    ${TORCH_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(${target} PRIVATE "${TORCH_LIBRARIES}")
  list(APPEND EXAMPLE_EXECUTABLE_TARGETS ${target})
  set(EXAMPLE_EXECUTABLE_TARGETS "${EXAMPLE_EXECUTABLE_TARGETS}" PARENT_SCOPE)
endfunction()

add_library(mujoco_rl_training SHARED
  envs/src/PendulumEnv.cpp
  envs/src/DoublePendulumEnv.cpp
)
target_mujoco_include_dirs(mujoco_rl_training)
target_link_libraries(mujoco_rl_training
  mujoco_core::mujoco_core
)
ament_target_dependencies(mujoco_rl_training
  mujoco_core
)

set(RL_EXECUTABLE_TARGETS)
add_rl_executable(pendulum_rl_demo src/pendulum_rl_demo.cpp)
add_rl_executable(pendulum_random_search src/pendulum_random_search.cpp)
add_rl_executable(double_pendulum_random_search src/double_pendulum_random_search.cpp)
add_rl_executable(pendulum_reinforce src/pendulum_reinforce.cpp)
add_rl_executable(pendulum_vpg src/pendulum_vpg.cpp)
add_rl_executable(double_pendulum_vpg src/double_pendulum_vpg.cpp)

add_rl_viewer_executable(pendulum_rl_visual_demo src/pendulum_rl_env_visualize.cpp)
add_rl_viewer_executable(double_pendulum_rl_visual_demo src/double_pendulum_rl_visual_demo.cpp)
add_rl_viewer_executable(pendulum_policy_rollout src/pendulum_policy_rollout.cpp)
add_rl_viewer_executable(double_pendulum_policy_rollout src/double_pendulum_policy_rollout.cpp)

set(EXAMPLE_EXECUTABLE_TARGETS)
add_torch_example(torch_tensor_basics examples/torch_tensor_basics.cpp)
add_torch_example(torch_autograd_basics examples/torch_autograd_basics.cpp)
add_torch_example(torch_manual_grad_descent examples/torch_manual_grad_descent.cpp)
add_torch_example(torch_linear_regression_with_opt examples/torch_linear_regression_with_opt.cpp)
add_torch_example(torch_mlp_critic examples/torch_mlp_critic.cpp)
add_torch_example(torch_gaussian_actor examples/torch_gaussian_actor.cpp)
add_torch_example(torch_gaussian_log_prob examples/torch_gaussian_log_prob.cpp)

install(
  TARGETS mujoco_rl_training
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS
    ${RL_EXECUTABLE_TARGETS}
    ${EXAMPLE_EXECUTABLE_TARGETS}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    include/
    envs/include/
  DESTINATION include
)

ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  # Formatting is enforced by repository clang-format/pre-commit, not ROS uncrustify.
  set(ament_cmake_uncrustify_FOUND TRUE)
  # Avoid network/schema lookups during hermetic pixi/CI test runs.
  set(ament_cmake_xmllint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
