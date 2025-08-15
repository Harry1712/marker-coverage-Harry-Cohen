cmake_minimum_required(VERSION 3.16)
project(marker_coverage CXX)

# ---- Compiler setup ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Warnings utiles (GCC/Clang)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
elseif (MSVC)
  add_compile_options(/W4)
endif()

# --- DEBUG_IMGPROC toggle (OFF by default) ---
option(DEBUG_IMGPROC "Enable OpenCV imshow debug windows" OFF)
if (DEBUG_IMGPROC)
  add_compile_definitions(DEBUG_IMGPROC)
endif()

# ---- OpenCV ----
find_package(OpenCV REQUIRED core imgproc highgui)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "DEBUG_IMGPROC: ${DEBUG_IMGPROC}")

# ---- Executable ----
add_executable(marker_coverage
  src/main.cpp
  src/ImageProc.cpp
)
target_include_directories(marker_coverage PRIVATE src)
target_link_libraries(marker_coverage PRIVATE ${OpenCV_LIBS})

# ---- Tests  ----
add_executable(test_marker_coverage
  test/test.cpp
  src/ImageProc.cpp
)
target_include_directories(test_marker_coverage PRIVATE src)
target_link_libraries(test_marker_coverage PRIVATE ${OpenCV_LIBS})

include(CTest)
if(BUILD_TESTING)
  add_test(NAME MarkerCoverageTest COMMAND test_marker_coverage)
endif()

set_target_properties(marker_coverage test_marker_coverage
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
