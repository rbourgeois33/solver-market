void SolverMarketOutput(const std::chrono::milliseconds& SolverMarketSetupTime, const std::chrono::milliseconds& SolverMarketSolveTime){
    
    auto out_solve = std::chrono::duration_cast<std::chrono::duration<double>>(SolverMarketSolveTime);
    auto out_setup = std::chrono::duration_cast<std::chrono::duration<double>>(SolverMarketSetupTime);


    std::cout << "\\---- Solver market output ----//\n";
    std::cout << "Setup time: " << out_setup.count() << std::setprecision(5) <<" s\n";
    std::cout << "Solve time: " << out_solve.count()*1000 << std::setprecision(5) <<" ms\n";
}