cmake_minimum_required(VERSION 3.22)
project(FrameKeeperRT LANGUAGES CXX)  # CUDA is enabled conditionally below

# ---------------- Basics ----------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Detect WSL (handy for headless paths or conditional code)
set(IS_WSL 0)
if(EXISTS "/proc/version")
  file(READ "/proc/version" _wslver)
  string(TOLOWER "${_wslver}" _wslver_lc)
  if(_wslver_lc MATCHES "microsoft")
    set(IS_WSL 1)
    message(STATUS "WSL environment detected")
    add_compile_definitions(FRAMEKEEPER_WSL=1)
  endif()
endif()

# Build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ENABLE_ASAN  "Enable AddressSanitizer" OFF)
option(ENABLE_LTO   "Enable link-time optimization" ON)
option(ENABLE_TESTS "Build tests in tests/ if present" ON)

# ---------------- CUDA (required) ----------------
# Try to locate nvcc. User can also pass -DCMAKE_CUDA_COMPILER=/path/to/nvcc
if(NOT CMAKE_CUDA_COMPILER)
  find_program(CMAKE_CUDA_COMPILER nvcc HINTS /usr/local/cuda/bin ENV PATH)
endif()

if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "CUDA is required but nvcc was not found. Please install CUDA Toolkit or set CMAKE_CUDA_COMPILER.")
endif()

# Set arch list BEFORE enabling CUDA language to avoid empty-arch error.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Safe defaults for modern GPUs; adjust if you want fewer.
  set(CMAKE_CUDA_ARCHITECTURES "86;89")
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
message(STATUS "CUDA enabled with nvcc: ${CMAKE_CUDA_COMPILER}")

# ---------------- Toolchain / QoL ----------------
if(ENABLE_ASAN AND (CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo"))
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

if(ENABLE_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_msg)
  if(_ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "LTO not supported: ${_ipo_msg}")
  endif()
endif()

# ---------------- Dependencies ----------------
include(FetchContent)
find_package(Threads REQUIRED)

# CUDA Toolkit targets (required)
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit found: ${CUDAToolkit_VERSION}")

# OpenCV (required)
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio dnn)
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
if(OpenCV_VERSION VERSION_GREATER_EQUAL "4.5.0")
  message(STATUS "OpenCV DNN module available for ML inference")
endif()

# TensorRT (required)
set(TENSORRT_ROOT "" CACHE PATH "Root of TensorRT")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} /usr/include /usr/include/x86_64-linux-gnu /usr/local/include
  PATH_SUFFIXES include nvinfer)
find_library(NVINFER_LIB nvinfer
  HINTS ${TENSORRT_ROOT} /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(NVONNXPARSER_LIB nvonnxparser
  HINTS ${TENSORRT_ROOT} /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB)
  message(FATAL_ERROR "TensorRT is required but not found. Please install TensorRT or set TENSORRT_ROOT.")
endif()

add_library(tensorrt::nvinfer UNKNOWN IMPORTED)
set_target_properties(tensorrt::nvinfer PROPERTIES
  IMPORTED_LOCATION "${NVINFER_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}")
if(NVONNXPARSER_LIB)
  add_library(tensorrt::nvonnxparser UNKNOWN IMPORTED)
  set_target_properties(tensorrt::nvonnxparser PROPERTIES
    IMPORTED_LOCATION "${NVONNXPARSER_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "TensorRT nvonnxparser library is required but not found.")
endif()
message(STATUS "TensorRT found: include=${TENSORRT_INCLUDE_DIR}, nvinfer=${NVINFER_LIB}")

# Header-only libs via FetchContent (fast, reproducible)
# CLI11
FetchContent_Declare(CLI11 GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git GIT_TAG v2.4.2)
FetchContent_MakeAvailable(CLI11)
# nlohmann::json
FetchContent_Declare(JSON URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(JSON)
# spdlog
FetchContent_Declare(spdlog GIT_REPOSITORY https://github.com/gabime/spdlog.git GIT_TAG v1.13.0)
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spdlog)
# httplib (cpp-httplib)
FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.15.3)
FetchContent_MakeAvailable(httplib)
# yaml-cpp (prefer system; else fetch)
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
  FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG yaml-cpp-0.8.0)
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(yaml-cpp)
endif()

# ---------------- Sources / Targets ----------------
# Include common C++ extensions and CUDA.
file(GLOB_RECURSE APP_SOURCES CONFIGURE_DEPENDS
  "src/*.cc" "src/*.cpp" "src/*.cxx" "src/*.cu"
)
file(GLOB_RECURSE APP_HEADERS CONFIGURE_DEPENDS
  "include/*.h" "include/*.hpp" "include/*.hh"
)

if(APP_SOURCES)
  add_executable(framekeeper-rt ${APP_SOURCES} ${APP_HEADERS})
  target_include_directories(framekeeper-rt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

  # Per-target warnings (don’t pollute third-party deps)
  target_compile_options(framekeeper-rt PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
    # For CUDA we must split tokens:
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag_suppress=esa_on_defaulted_function_ignored>
  )

  # Core links (all required)
  target_link_libraries(framekeeper-rt PRIVATE
    Threads::Threads
    CLI11::CLI11
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    yaml-cpp
    httplib::httplib
    CUDA::cudart
    ${OpenCV_LIBS}
    tensorrt::nvinfer
    tensorrt::nvonnxparser
  )

  # WSL-specific configuration
  if(IS_WSL)
    target_compile_definitions(framekeeper-rt PRIVATE HEADLESS=1)
  endif()

  # RPATH for TRT at runtime
  get_filename_component(_trt_libdir "${NVINFER_LIB}" DIRECTORY)
  set_property(TARGET framekeeper-rt PROPERTY BUILD_RPATH "${_trt_libdir}")

  # Speed flags for Release
  target_compile_options(framekeeper-rt PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -DNDEBUG>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 -DNDEBUG>
  )

  message(STATUS "Target 'framekeeper-rt' configured for ${CMAKE_BUILD_TYPE}.")
else()
  message(WARNING "No sources in src/. Add code then re-run CMake.")
endif()

# ---------------- Tests ----------------
if(ENABLE_TESTS)
  file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "tests/*.cc" "tests/*.cpp" "tests/*.cxx")
  if(TEST_SOURCES)
    enable_testing()
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0)
    FetchContent_MakeAvailable(googletest)

    add_executable(unit_tests ${TEST_SOURCES})
    target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(TARGET framekeeper-rt)
      target_link_libraries(unit_tests PRIVATE framekeeper-rt)
    endif()
    target_link_libraries(unit_tests PRIVATE GTest::gtest_main)
    add_test(NAME unit_tests COMMAND unit_tests)
  endif()
endif()

# ---------------- Convenience ----------------
if(TARGET framekeeper-rt)
  add_custom_target(run
    COMMAND $<TARGET_FILE:framekeeper-rt> --help
    COMMENT "Run the demo binary (or show help).")
else()
  add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E echo "No 'framekeeper-rt' target yet. Add sources under src/ and re-configure."
    COMMENT "Placeholder 'run' target (no sources found).")
endif()

add_custom_target(format
  COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.h" | xargs clang-format -i
  COMMENT "Formatting C++ and CUDA source files with clang-format"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(format-check
  COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.h" | xargs clang-format --dry-run --Werror
  COMMENT "Check if C++ and CUDA source files are properly formatted"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
