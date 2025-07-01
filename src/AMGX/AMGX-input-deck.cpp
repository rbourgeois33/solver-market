#include <iostream>
#include <string>
#include <amgx_c.h>
#include <cstring>

#include "solver-market-csr-matrix.hpp"
#include "solver-market-vector.hpp"
#include <chrono>
#include <solver-market-output.h>


//AMGX_RC is AMGX Return Code, all AMGX functions output one
//This function helps decoding it
void check_AMGX_error(AMGX_RC rc, const char *msg) 
{
    if (rc != AMGX_RC_OK) {
        char err_string[256];
        AMGX_get_error_string(rc, err_string, sizeof(err_string));
        std::cerr << "Error: " << msg << " - " << err_string << std::endl;
        exit(EXIT_FAILURE);
    } 
}


int main(int argc, char* argv[])
{
    Kokkos::initialize();
    {
    std::string matrix_file;
    std::string rhs_file;
    std::string config_file;

    // 1. Parse input arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--matrix=", 0) == 0) {
            matrix_file = arg.substr(9);  // after "--matrix="
        } else if (arg.rfind("--rhs=", 0) == 0) {
            rhs_file = arg.substr(6);  // after "--rhs="
        } else if (arg.rfind("--config=", 0) == 0) {
            config_file = arg.substr(9);  // after "--rhs="
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return EXIT_FAILURE;
        }
    }

    if (matrix_file.empty() || config_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " --matrix=<matrix_file.mtx> --rhs=<rhs_file.mtx> (optional) --config=<config_file.mtx> " << std::endl;
        return EXIT_FAILURE;
    }

    // SolverMarket timers
    std::chrono::milliseconds SolverMarketSetupTime, SolverMarketSolveTime;

    //RC object for error handling
    AMGX_RC rc;

    // 2. Initialize AMGX
    rc = AMGX_initialize();
    check_AMGX_error(rc, "AMGX_initialize error:");


    //Capture and print AMGX version:
    int major, minor;
    AMGX_get_api_version(&major, &minor);
    std::cout << "Using AMGX API version: " << major << "." << minor<< std::endl;

    // 3. Create AMGX configuration from file
    // Load configuration from JSON file
    AMGX_config_handle config = nullptr;
    rc = AMGX_config_create_from_file(&config, config_file.c_str());
    check_AMGX_error(rc, "AMGX_config_create error:");

    // 4. Create AMGX resources
    AMGX_resources_handle rsrc = NULL;
    AMGX_resources_create_simple(&rsrc, config);
    check_AMGX_error(rc, "AMGX_resources_create_simple:");

    //Choose mode
    //d: Double precision
    //DD: matrix and vector are fully distributed 
    //I: index are 32 bits
    auto mode = AMGX_mode_dDDI;

    // 5. Create solver object
    AMGX_solver_handle solver = NULL;
    rc = AMGX_solver_create(&solver, rsrc, mode, config);
    check_AMGX_error(rc, "AMGX_solver_create:");

    // 6. Create matrix and vectors
    AMGX_matrix_handle A = NULL;
    AMGX_vector_handle x = NULL;
    AMGX_vector_handle b = NULL;

    AMGX_matrix_create(&A, rsrc, mode);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);

    // 7. Read system from .mtx file
    auto matrix =  SolverMarketCSRMatrix<double, int>();
    auto result = matrix.read_matrix_market_file(matrix_file, SolverMarketCSRMatrixFull);

    matrix.send_to_device();
    AMGX_matrix_upload_all(A,
          matrix.get_n(), 
          matrix.get_nnz(), 1, 1, matrix.get_device_offsets_pointer(), matrix.get_device_columns_pointer(), matrix.get_device_values_pointer(), 0);
    
    
    if (rhs_file.empty()){
    std::cout <<"No vector b given, filling with 1"<<std::endl;
    
    auto vector_b =  SolverMarketVector<double, int>(matrix.get_n(), 1.0);
    AMGX_vector_upload(b, matrix.get_n(), 1, vector_b.get_host_values_pointer());}
    else{
    auto vector_b =  SolverMarketVector<double, int>(rhs_file);
    AMGX_vector_upload(b, matrix.get_n(), 1, vector_b.get_host_values_pointer());}
    
    auto vector_x =  SolverMarketVector<double, int>(matrix.get_n(), 0.0);
    AMGX_vector_upload(x, matrix.get_n(), 1, vector_x.get_host_values_pointer());

    //SolverMarket: time setup
    auto start = std::chrono::high_resolution_clock::now();
    // 8. Setup the solver (analysis phase)
    rc = AMGX_solver_setup(solver, A);
    check_AMGX_error(rc, "AMGX_solver_setup:");
    auto end = std::chrono::high_resolution_clock::now();
    SolverMarketSetupTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    //SolverMarket time solve
    start = std::chrono::high_resolution_clock::now();
    // 9. Solve the system
    rc = AMGX_solver_solve(solver, b, x);
    end = std::chrono::high_resolution_clock::now();
    SolverMarketSolveTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    check_AMGX_error(rc, "AMGX_solver_solve:");
    //No need to print stuff: AMGX handles it (optional in config/json file)

    SolverMarketOutput(SolverMarketSetupTime, SolverMarketSolveTime, rc==0, argc, argv);

    // 10. Clean up and shut down
    AMGX_solver_destroy(solver);
    AMGX_matrix_destroy(A);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(config);
    
    // Finalize AMGX
    AMGX_finalize();
    std::cout << "AMGX solve complete." << std::endl;
    }
    Kokkos::finalize();
    return EXIT_SUCCESS;
}