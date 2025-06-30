template<typename _TYPE_, typename _ITYPE_>
int SolverMarketVector<_TYPE_, _ITYPE_>::send_to_device(){

    if (not(is_allocated_)){
        std::cout<<"[Error][SolverMarket][CsrVector][send_to_device] You want to send to device a CSR vector that has not been allocated\n";
        return 1;
    }
    Kokkos::deep_copy(values_d_, values_h_);

    std::cout << "[Info][SolverMarket][CsrVector][send_to_device] Values successfuly sent to device\n";

    return 0;
  }

template<typename _TYPE_, typename _ITYPE_>
int SolverMarketVector<_TYPE_, _ITYPE_>::allocate(const _ITYPE_ n){
    n_ = n;
    values_h_ = HostView<_TYPE_>("values_h_", n);
    values_d_ = DeviceView<_TYPE_>("values_d_", n);
    std::cout << "[Info][SolverMarket][CsrVector][allocate] Successfuly allocated on host and device\n";


    is_allocated_ = true;
    return 0;
  }

template<typename _TYPE_, typename _ITYPE_>
int SolverMarketVector<_TYPE_, _ITYPE_>::read_matrix_market_file(std::string filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[Error][SolverMarket][Vector][read_from_file] Could not open file: " << filename << "\n";
        return MtxReaderErrorFileNotFound;
    }else{
        std::cout << "[Info][SolverMarket][CsrMatrix][read_from_file] Reading file "<< filename << std::endl;
    }

    std::string line;
    bool foundHeader = false;
    bool foundSize = false;
    int declared_nnz = 0;
    int file_line_count = 0;
    int nrows = 0, ncols = 0;

    std::vector<std::pair<_ITYPE_, _TYPE_>> entries;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (!foundHeader) {
            if (line.rfind("%%MatrixMarket", 0) == 0) {
                std::istringstream header(line);
                std::string banner, object, format, field, symmetry;
                header >> banner >> object >> format >> field >> symmetry;

                if (object != "matrix" || format != "coordinate") {
                    std::cerr << "[Error][SolverMarket][Vector][read_from_file] Only 'matrix coordinate' format supported.\n";
                    return MtxReaderUnsupportedObject;
                }

                if (symmetry == "general"){}
                else if (symmetry == "symmetric"){}
                else {
                    std::cerr << "[Error][SolverMarket][Vector][read_from_file] Unsupported matrix type: " << symmetry << "\n";
                    return MtxReaderUnsupportedMatrixType;
                }

                foundHeader = true;
                continue;
            }
        }

        std::istringstream lineData(line);
        if (!foundSize) {
            lineData >> nrows >> ncols >> declared_nnz;

            if (ncols != 1) {
                std::cerr << "[Error][SolverMarket][Vector][read_from_file] Not a vector (ncols = " << ncols << ")\n";
                return MtxReaderNotAVector;
            }

            if (nrows != declared_nnz) {
                std::cerr << "[Error][SolverMarket][Vector][read_from_file] Not a vector (nnz = " << nrows << " != nrows ="<< declared_nnz <<")\n";
                return MtxReaderNotAVector;
            }

            foundSize = true;
        } else {
            _ITYPE_ i;
            _ITYPE_ j;
            _TYPE_ val;

            lineData >> i >> j >> val;
            i -= 1; // 1-based to 0-based
            if (i < 0 || i >= nrows) {
                std::cerr << "[Error][SolverMarket][Vector][read_from_file] Invalid row index " << i << "\n";
                return MtxReaderErrorOutOfBoundRowIndex;
            }

            if (j != 1) {
                std::cerr << "[Error][SolverMarket][Vector][read_from_file] Invalid col index != 1 " << j << "\n";
                return MtxReaderErrorOutOfBoundColIndex;
            }

            entries.emplace_back(i, val);
            file_line_count++;
        }
    }

    file.close();

    if (!foundHeader || !foundSize) {
        std::cerr << "[Error][SolverMarket][Vector][read_from_file] Invalid Matrix Market header or size line.\n";
        return MtxReaderWrongHeaderOrNoHeader;
    }

    n_ = nrows;

    if (allocate(nrows)) {
        std::cerr << "[Error][SolverMarket][Vector][read_from_file] Memory allocation failed.\n";
        return MtxReaderErrorFileMemAllocFailed;
    }

    for (auto& [i, val] : entries) {
        values_h_(i) = val;
    }

    std::cout << "[Info][SolverMarket][Vector][read_from_file] Successfully read vector with " << nrows
              << " entries, " << file_line_count << " non-zeros\n";
    return MtxReaderSuccess;
}