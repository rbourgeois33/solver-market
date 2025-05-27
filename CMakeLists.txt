cmake_minimum_required(VERSION 3.16)
project(SolverMarket LANGUAGES CXX)

# Add Trilinos as a submodule (already cloned)
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/external/trilinos/install ${CMAKE_PREFIX_PATH})
# Search for Trilinos
find_package(Trilinos REQUIRED)

# Print what's found
message(STATUS "Trilinos found: ${Trilinos_DIR}")
message(STATUS "Trilinos version: ${Trilinos_VERSION}")
message(STATUS "Trilinos packages: ${Trilinos_PACKAGE_LIST}")
message(STATUS "Trilinos TPLs:     ${Trilinos_TPL_LIST}")

# Compiler and flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

# Include Trilinos and TPLs
include_directories(${Trilinos_INCLUDE_DIRS})
include_directories(${Trilinos_TPL_INCLUDE_DIRS})
link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})

# Your executable
include_directories(${CMAKE_SOURCE_DIR}/external/trilinos/packages/muelu/test/scaling)
include_directories(${CMAKE_SOURCE_DIR}/external/trilinos/packages/muelu/test/unit_tests)
add_executable(muelu_input_deck input-decks/muelu-input-deck.cpp)

# Link against Trilinos and its TPLs
target_link_libraries(muelu_input_deck
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
)