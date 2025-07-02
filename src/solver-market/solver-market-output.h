#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <string>

void SolverMarketOutput(const std::chrono::milliseconds& SolverMarketSetupTime, 
                        const std::chrono::milliseconds& SolverMarketSolveTime,
                        bool success,
                        int argc, char *argv[]) {
    
    auto out_solve = std::chrono::duration_cast<std::chrono::duration<double>>(SolverMarketSolveTime);
    auto out_setup = std::chrono::duration_cast<std::chrono::duration<double>>(SolverMarketSetupTime);

    std::cout << "\n \\---- Solver Market output ----/\n\n";
    std::cout << "input: ";
    std::ostringstream input;
    for (int i = 0; i < argc; i++) {
        std::cout << argv[i] << " ";
        input << argv[i] << " ";
    }

    std::cout << "\n";
    std::cout << "Success: " << success << "\n";
    std::cout << "Setup time: " << std::setprecision(6) << out_setup.count() << " s\n";
    std::cout << "Solve time: " << std::setprecision(6) << out_solve.count() * 1000 << " ms\n\n";

    std::cout << "\n \\-----------------------------/\n";

    // --- Write to file (append mode)
    std::ofstream outFile("solver_output.log", std::ios::app);
    if (outFile.is_open()) {
        outFile << input.str() << " "
                << std::fixed << std::setprecision(6)
                << success<< " "
                << out_setup.count() << " "
                << out_solve.count() * 1000 << "\n";
        outFile.close();
    } else {
        std::cerr << "Error: Could not open solver_output.log for writing.\n";
    }
}