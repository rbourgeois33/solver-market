cmake_minimum_required(VERSION 3.16)
project(SolverMarket LANGUAGES CXX)

# C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

# ===============================
# 🔘 Build Options
# ===============================
option(BUILD_MUELU_INPUT_DECK "Build muelu input deck example" ON)
option(BUILD_AMGX_INPUT_DECK "Build AMGX input deck example" ON)
option(BUILD_UNIT_TESTS "Build unit tests" ON)

# ===============================
# 🔍 Try to Find Trilinos
# ===============================
# Try to find Trilinos
find_package(Trilinos REQUIRED)

# Verify Trilinos was found
if(NOT Trilinos_FOUND)
    message(FATAL_ERROR "Trilinos not found! Please build and install Trilinos with Kokkos enabled.")
endif()

# Verify Kokkos is available within Trilinos
if(NOT "Kokkos" IN_LIST Trilinos_PACKAGE_LIST)
    message(FATAL_ERROR "Trilinos found, but does not include the Kokkos package! Please enable Kokkos in your Trilinos build.")
endif()

# Debug output
message(STATUS "✅ Trilinos found")
message(STATUS "Trilinos version: ${Trilinos_VERSION}")
message(STATUS "Trilinos packages: ${Trilinos_PACKAGE_LIST}")
message(STATUS "Trilinos TPLs:     ${Trilinos_TPL_LIST}")

# ===============================
# 🔧 Build muelu_input_deck
# ===============================
if(BUILD_MUELU_INPUT_DECK AND HAVE_TRILINOS)
    add_executable(muelu_input_deck src/muelu/muelu-input-deck.cpp)

    # Output binary location
    set_target_properties(muelu_input_deck PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/input-decks/
    )

    # Include paths
    target_include_directories(muelu_input_deck PRIVATE
        ${Trilinos_INCLUDE_DIRS}
        ${Trilinos_TPL_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/external/trilinos/packages/muelu/test/scaling
        ${CMAKE_SOURCE_DIR}/external/trilinos/packages/muelu/test/unit_tests
        ${CMAKE_SOURCE_DIR}/src/solver-market
    )

    # Link Trilinos and MPI
    find_package(MPI REQUIRED)
    target_link_libraries(muelu_input_deck
        PRIVATE
        MPI::MPI_CXX
        ${Trilinos_LIBRARIES}
        ${Trilinos_TPL_LIBRARIES}
    )

    # utils for muelu
    add_executable(ascii2binary ${CMAKE_SOURCE_DIR}/external/trilinos/packages/muelu/utils/matrix/ascii2binary.cpp)
    set_target_properties(ascii2binary PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/utils/
    )
endif()

# ===============================
# 🔧 Build AMGX_input_deck
# ===============================
if(BUILD_AMGX_INPUT_DECK)
    # Add AMGX headers and libraries
    include_directories("${CMAKE_SOURCE_DIR}/external/AMGX/include")
    link_directories("${CMAKE_SOURCE_DIR}/external/AMGX/build")

    add_executable(AMGX_input_deck src/AMGX/AMGX-input-deck.cpp)

    target_include_directories(AMGX_input_deck PRIVATE
        ${CMAKE_SOURCE_DIR}/src/solver-market
    )

    # Output binary location
    set_target_properties(AMGX_input_deck PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/input-decks/
    )

    # Link to Trilinos
    target_include_directories(AMGX_input_deck PRIVATE
        ${Trilinos_INCLUDE_DIRS}
    )
    target_link_libraries(AMGX_input_deck
        PRIVATE
        ${Trilinos_LIBRARIES} amgxsh
    )


endif()

# ===============================
# 🔧 Unit Tests with GTest + kokkos from trilinos
# ===============================
if(BUILD_UNIT_TESTS)
    # Fetch GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Add unit test executable
    add_executable(unit-test-solver-market-reader tests/unit-test-solver-market-reader.cpp)

    # Output binary location
    set_target_properties(unit-test-solver-market-reader PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/
    )

    # Common includes
    target_include_directories(unit-test-solver-market-reader PRIVATE
        ${CMAKE_SOURCE_DIR}/src/solver-market
    )

    # Link to Trilinos (for kokkos)
    target_include_directories(unit-test-solver-market-reader PRIVATE
        ${Trilinos_INCLUDE_DIRS}
    )
    target_link_libraries(unit-test-solver-market-reader
        PRIVATE
        ${Trilinos_LIBRARIES}
    )

    # Link GTest
    target_link_libraries(unit-test-solver-market-reader
        PRIVATE
        GTest::gtest
        GTest::gtest_main
    )
endif()

# ===============================
# 🔍 Build Summary
# ===============================
message(STATUS "")
message(STATUS "======== Build Configuration ========")
message(STATUS "BUILD_MUELU_INPUT_DECK: ${BUILD_MUELU_INPUT_DECK}")
message(STATUS "BUILD_AMGX_INPUT_DECK:  ${BUILD_AMGX_INPUT_DECK}")
message(STATUS "BUILD_UNIT_TESTS:       ${BUILD_UNIT_TESTS}")
message(STATUS "=====================================")
