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


# Point CMake to AMGX headers
include_directories("${CMAKE_SOURCE_DIR}/external/AMGX/include")

# # Point CMake to AMGX libraries
link_directories("${CMAKE_SOURCE_DIR}/external/AMGX/build")

# # Create the executable from your source file
add_executable(amgx_input_deck src/amgx/amgx-input-deck.cpp)

# # Link against the AMGX shared library (AMGXsh).
target_link_libraries(amgx_input_deck amgxsh)



include(FetchContent)

# Download GTest if not already available
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Download and make GTest targets
FetchContent_MakeAvailable(googletest)


add_executable(unit-test-solver-market-csr-matrix src/unit-test-solver-market-csr-matrix.cpp)

# Include dirs (Trilinos includes Kokkos)
target_include_directories(unit-test-solver-market-csr-matrix PRIVATE
    ${Trilinos_INCLUDE_DIRS}
    ${Trilinos_TPL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)

# Link Trilinos, MPI, and GTest
target_link_libraries(unit-test-solver-market-csr-matrix
    PRIVATE
    MPI::MPI_CXX
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
)