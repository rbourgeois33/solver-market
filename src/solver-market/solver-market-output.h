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

    std::cout << "\\---- Solver market output ----/\n";
    std::cout << "input: ";
    for (int i = 0; i < argc; i++) {
        std::cout << argv[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Success: " << success << "\n";
    std::cout << "Setup time: " << std::setprecision(5) << out_setup.count() << " s\n";
    std::cout << "Solve time: " << std::setprecision(5) << out_solve.count() * 1000 << " ms\n";

    // --- Extract just the XML filename
    std::string xmlFileName = "unknown.xml";
    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        auto pos = arg.find("--xml=");
        if (pos != std::string::npos) {
            std::string fullPath = arg.substr(pos + 6); // Skip "--xml="
            size_t lastSlash = fullPath.find_last_of("/\\");
            if (lastSlash != std::string::npos) {
                xmlFileName = fullPath.substr(lastSlash + 1);
            } else {
                xmlFileName = fullPath;
            }
            break;
        }
    }

    // --- Write to file (append mode)
    std::ofstream outFile("solver_output.log", std::ios::app);
    if (outFile.is_open()) {
        outFile << xmlFileName << " "
                << std::fixed << std::setprecision(5)
                << success<< " "
                << out_setup.count() << " "
                << out_solve.count() * 1000 << "\n";
        outFile.close();
    } else {
        std::cerr << "Error: Could not open solver_output.log for writing.\n";
    }
}