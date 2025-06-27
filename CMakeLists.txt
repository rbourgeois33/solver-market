cmake_minimum_required(VERSION 3.16)
project(SolverMarket LANGUAGES CXX)

# C++ settings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

# ===============================
# 🔘 Build Options
# ===============================
option(BUILD_MUELU_INPUT_DECK "Build muelu input deck example" ON)
option(BUILD_AMGX_INPUT_DECK "Build amgx input deck example" ON)
option(BUILD_UNIT_TESTS "Build unit tests" ON)

# ===============================
# 🔍 Try to Find Trilinos
# ===============================
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/external/trilinos/install ${CMAKE_PREFIX_PATH})
find_package(Trilinos QUIET)

if(Trilinos_FOUND)
    message(STATUS "Trilinos found")
    set(HAVE_TRILINOS TRUE)
    message(STATUS "Trilinos version: ${Trilinos_VERSION}")
    message(STATUS "Trilinos packages: ${Trilinos_PACKAGE_LIST}")
    message(STATUS "Trilinos TPLs:     ${Trilinos_TPL_LIST}")
else()
    message(WARNING "Trilinos not found! Attempting to fetch and build Kokkos for unit tests...")

    # Download Kokkos as fallback (for unit tests)
    include(FetchContent)
    FetchContent_Declare(
        kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG 4.2.00
    )
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Serial backend")
    set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Disable Kokkos tests")
    FetchContent_MakeAvailable(kokkos)
    set(HAVE_TRILINOS FALSE)
endif()

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
    )

    # Link Trilinos and MPI
    find_package(MPI REQUIRED)
    target_link_libraries(muelu_input_deck
        PRIVATE
        MPI::MPI_CXX
        ${Trilinos_LIBRARIES}
        ${Trilinos_TPL_LIBRARIES}
    )
endif()

# ===============================
# 🔧 Build amgx_input_deck
# ===============================
if(BUILD_AMGX_INPUT_DECK)
    # Add AMGX headers and libraries
    include_directories("${CMAKE_SOURCE_DIR}/external/AMGX/include")
    link_directories("${CMAKE_SOURCE_DIR}/external/AMGX/build")

    add_executable(amgx_input_deck src/amgx/amgx-input-deck.cpp)

    # Output binary location
    set_target_properties(amgx_input_deck PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/input-decks/
    )

    # Link AMGX shared lib
    target_link_libraries(amgx_input_deck amgxsh)
endif()

# ===============================
# 🔧 Unit Tests with GTest + Kokkos fallback
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
    add_executable(unit-test-solver-market-csr-matrix tests/unit-test-solver-market-csr-matrix.cpp)

    # Output binary location
    set_target_properties(unit-test-solver-market-csr-matrix PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/
    )

    # Common includes
    target_include_directories(unit-test-solver-market-csr-matrix PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    # Link either Trilinos or fallback Kokkos
    if(HAVE_TRILINOS)
        target_include_directories(unit-test-solver-market-csr-matrix PRIVATE
            ${Trilinos_INCLUDE_DIRS}
            ${Trilinos_TPL_INCLUDE_DIRS}
        )
        target_link_libraries(unit-test-solver-market-csr-matrix
            PRIVATE
            MPI::MPI_CXX
            ${Trilinos_LIBRARIES}
            ${Trilinos_TPL_LIBRARIES}
        )
    else()
        target_link_libraries(unit-test-solver-market-csr-matrix PRIVATE kokkos)
    endif()

    # Link GTest
    target_link_libraries(unit-test-solver-market-csr-matrix
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
message(STATUS "Trilinos available:     ${HAVE_TRILINOS}")
message(STATUS "=====================================")