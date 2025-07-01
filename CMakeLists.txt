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
# 🔍 Getting Trilinos
# ===============================

# Trilinos provides both kokkos and solvers that we use in this repo
# Moreover, Kokkos is at the core of our solver-market routines.
# Therefore, even if we are not using trilinos's solver, we might still want kokkos.

set(TRILINOS_INSTALL_DIR ${CMAKE_SOURCE_DIR}/external/trilinos/install)
set(TRILINOS_KOKKOS_ONLY_INSTALL_DIR ${CMAKE_SOURCE_DIR}/external/trilinos/install-kokkos-only)

# Decide which Trilinos install directory to use
if(EXISTS "${TRILINOS_INSTALL_DIR}/lib/cmake/Trilinos")
  set(SELECTED_TRILINOS_DIR ${TRILINOS_INSTALL_DIR})
  set(IS_KOKKOS_ONLY FALSE)
  message(STATUS "✅ Found full Trilinos at ${TRILINOS_INSTALL_DIR}")
elseif(EXISTS "${TRILINOS_KOKKOS_ONLY_INSTALL_DIR}/lib/cmake/Trilinos")
  message(STATUS "❌ No Trilinos at ${TRILINOS_INSTALL_DIR}")
  set(SELECTED_TRILINOS_DIR ${TRILINOS_KOKKOS_ONLY_INSTALL_DIR})
  set(IS_KOKKOS_ONLY TRUE)
  message(STATUS "✅ Found Kokkos-only Trilinos at ${TRILINOS_KOKKOS_ONLY_INSTALL_DIR}")
else()
  message(FATAL_ERROR "❌ No Trilinos installation found in either '${TRILINOS_INSTALL_DIR}' or '${TRILINOS_KOKKOS_ONLY_INSTALL_DIR}'")
endif()

# Set CMAKE_PREFIX_PATH and find Trilinos
set(CMAKE_PREFIX_PATH ${SELECTED_TRILINOS_DIR} ${CMAKE_PREFIX_PATH})
find_package(Trilinos QUIET)
# Set Common paths for include and lib
set(Kokkos_INCLUDE_DIR ${SELECTED_TRILINOS_DIR}/include/kokkos)
set(Trilinos_LIB_DIR ${SELECTED_TRILINOS_DIR}/lib)

# Validate compatibility with build options
if(IS_KOKKOS_ONLY AND BUILD_MUELU_INPUT_DECK)
  message(FATAL_ERROR "MueLu input deck requested, but only a Kokkos-only install of Trilinos was found. Please build full Trilinos with MueLu enabled.")
endif()

# Summary
message(STATUS "✅ Trilinos successfully configured from: ${SELECTED_TRILINOS_DIR}")
message(STATUS "Trilinos version: ${Trilinos_VERSION}")

# ===============================
# 🔧 Build muelu_input_deck
# ===============================
if(BUILD_MUELU_INPUT_DECK)
    add_executable(muelu_input_deck src/muelu/muelu-input-deck.cpp)

    # Output binary location
    set_target_properties(muelu_input_deck PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/input-decks/
    )

    # Include paths
    target_include_directories(muelu_input_deck PRIVATE
        ${Kokkos_INCLUDE_DIR} 
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

    # Output binary location
    set_target_properties(AMGX_input_deck PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/input-decks/
    )

    target_include_directories(AMGX_input_deck PRIVATE
        ${CMAKE_SOURCE_DIR}/src/solver-market
        ${Kokkos_INCLUDE_DIR} 
    )

    target_link_libraries(AMGX_input_deck
        PRIVATE
        ${Trilinos_LIBRARIES} amgxsh
        "${Trilinos_LIB_DIR}/libkokkoscore.so"
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
        ${Kokkos_INCLUDE_DIR} 
        ${Trilinos_INCLUDE_DIRS}
    )

    # Link GTest
    target_link_libraries(unit-test-solver-market-reader
        PRIVATE
        GTest::gtest
        GTest::gtest_main
        "${Trilinos_LIB_DIR}/libkokkoscore.so"
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
