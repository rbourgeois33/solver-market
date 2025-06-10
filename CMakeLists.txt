cmake_minimum_required(VERSION 3.16)
project(SolverMarket LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

# Find Trilinos
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/external/trilinos/install ${CMAKE_PREFIX_PATH})
find_package(Trilinos REQUIRED)
find_package(MPI REQUIRED)

message(STATUS "Trilinos version: ${Trilinos_VERSION}")
message(STATUS "Trilinos packages: ${Trilinos_PACKAGE_LIST}")
message(STATUS "Trilinos TPLs:     ${Trilinos_TPL_LIST}")

# Executable
add_executable(muelu_input_deck src/muelu/muelu-input-deck.cpp)

# Include paths
target_include_directories(muelu_input_deck PRIVATE
    ${Trilinos_INCLUDE_DIRS}
    ${Trilinos_TPL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/trilinos/packages/muelu/test/scaling
    ${CMAKE_SOURCE_DIR}/external/trilinos/packages/muelu/test/unit_tests
)

# Link Trilinos + MPI
target_link_libraries(muelu_input_deck
    PRIVATE
    MPI::MPI_CXX
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
)