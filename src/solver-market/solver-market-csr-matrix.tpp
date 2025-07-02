template<typename _TYPE_, typename _ITYPE_>
int SolverMarketCSRMatrix<_TYPE_, _ITYPE_>::send_to_device(){

    if (not(is_allocated_)){
        std::cout<<"[Error][SolverMarket][CsrMatrix][send_to_device] You want to send to device a CSR matrix that has not been allocated\n";
        return 1;
    }

    Kokkos::deep_copy(offsets_d_, offsets_h_);
    Kokkos::deep_copy(columns_d_, columns_h_);
    Kokkos::deep_copy(values_d_, values_h_);

    std::cout << "[Info][SolverMarket][CsrMatrix][send_to_device] Values successfuly sent to device\n";

    return 0;
  }

template<typename _TYPE_, typename _ITYPE_>
int SolverMarketCSRMatrix<_TYPE_, _ITYPE_>::allocate(const _ITYPE_ n, const _ITYPE_ nnz){
    n_ = n;
    nnz_ = nnz;

    offsets_h_ = HostView<_ITYPE_>("offsets_h_", n+1);
    columns_h_ = HostView<_ITYPE_>("columns_h_", nnz);
    values_h_ = HostView<_TYPE_>("values_h_", nnz);

    offsets_d_ = DeviceView<_ITYPE_>("offsets_d_", n+1);
    columns_d_ = DeviceView<_ITYPE_>("columns_d_", nnz);
    values_d_ = DeviceView<_TYPE_>("values_d_", nnz);

    std::cout << "[Info][SolverMarket][CsrMatrix][allocate] Successfuly allocated on host and device\n";


    is_allocated_ = true;
    return 0;
  }

template<typename _TYPE_, typename _ITYPE_>
int SolverMarketCSRMatrix<_TYPE_, _ITYPE_>::read_matrix_market_file(std::string filename, SolverMarketCSRMatrixView mview, SolverMarketCSRMatrixType mtype)
{

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[Error][SolverMarket][CsrMatrix][read_from_file] Could not open file" << filename << std::endl;
        return  MtxReaderErrorFileNotFound;
    }else{
        std::cout << "[Info][SolverMarket][CsrMatrix][read_from_file] Reading file "<< filename << std::endl;
    }

    std::string line;
    bool foundSize = false;
    bool foundHeader = false;
    bool justFoundHeader=false;
    bool found_lower = false, found_upper = false;
    int declared_nnz = 0;
    int file_line_count = 0;
    int n, nnz;
    SolverMarketCSRMatrixType read_type = SolverMarketCSRMatrixTypeNone;

    //Tuple for the mtw lines
    std::vector<std::tuple<int, int, _TYPE_>> entries;


    while (std::getline(file, line)) {
        //Skip header
        if (line.empty()) continue;

        //Skip comment after header iff present
        if (justFoundHeader && line[0]=='%'){
            justFoundHeader=false;
            continue;
        }


        // Parse MatrixMarket header
        if (!foundHeader) {
            if (line.rfind("%%MatrixMarket", 0) == 0) {
                std::istringstream header(line);
                std::string banner, object, format, field, symmetry;
                header >> banner >> object >> format >> field >> symmetry;

                if (object != "matrix" || format != "coordinate") {
                    std::cerr << "[Error][SolverMarket][CsrMatrix][read_from_file] Only 'matrix coordinate' format is supported.\n";
                    return MtxReaderUnsupportedObject;
                }

                if (symmetry == "general") read_type = SolverMarketCSRMatrixGeneral;
                else if (symmetry == "symmetric") read_type = SolverMarketCSRMatrixSymmetric;
                else {
                    std::cerr << "[Error][SolverMarket][CsrMatrix][read_from_file]  Unsupported matrix type: " << symmetry << "\n";
                    return MtxReaderUnsupportedMatrixType;
                }


                if (mtype == SolverMarketCSRMatrixTypeNone){
                    mtype_ = read_type;
                    std::cout << "[Info][SolverMarket][CsrMatrix][read_from_file] Matrix type read in mtx header: "<<read_type<<".\n";
                }else{
                    if (mtype_ == read_type)
                        std::cout << "[Info][SolverMarket][CsrMatrix][read_from_file] Matrix type read in mtx header: "<<read_type<<".\n";
                    else
                        std::cerr << "[Error][SolverMarket][CsrMatrix][read_from_file] Matrix type read in mtx header: "<<read_type<<" but mtx reader was called with type"<<mtype<<".\n";
                        return MtxReaderTypeReadIsNotTypeGiven;
                }

                foundHeader = true;
                justFoundHeader=true;
            }
            continue; // Skip other comment lines before header
        }
        //Get metadata
        std::istringstream lineData(line);
        if (!foundSize) {
            int n1;
            lineData >> n >> n1 >> declared_nnz;
            std::cout << "[Info][SolverMarket][CsrMatrix][read_from_file] n0= " << n << ", n1= " << n1 << ", nnz= " << declared_nnz << "\n";
            foundSize = true;
        //Get line
        } else {
            _ITYPE_ i, j;
            _TYPE_ val;
            lineData >> i >> j >> val;
            i -= 1; j -= 1;  // Convert from 1-based to 0-based

            //Append the tuple to entries (resize everytime so sub-optimal)
            entries.emplace_back(i, j, val);

            if (i > j) found_lower = true;
            if (i < j) found_upper = true;
            file_line_count+=1;
        }
    }

    if (not(foundHeader)){
        std::cerr << "[Error][SolverMarket][CsrMatrix][read_from_file] No header found in mtx file.\n";
        return MtxReaderWrongHeaderOrNoHeader;
    }

    file.close();
    nnz = entries.size();

    /* set view type */
    mview_ = mview;

    // Allocate memory
    int failed = allocate(n, nnz);
    if (failed) {
        std::cerr << "[Error][SolverMarket][CsrMatrix][read_from_file] Memory allocation failed\n";
        return MtxReaderErrorFileMemAllocFailed;
    }

    if (file_line_count != declared_nnz) {
        std::cerr << "[Error][SolverMarket][CsrMatrix][read_from_file] More elements in the mtx file than announced in the header\n";
        return MtxReaderErrorWrongNnz;
    }


    // Sort by row, then by column, 3rd argument is the compare operator, (row then column)
    std::sort(entries.begin(), entries.end(), [](auto& a, auto& b) {
        if (std::get<0>(a) == std::get<0>(b))
            return std::get<1>(a) < std::get<1>(b);
        return std::get<0>(a) < std::get<0>(b);
    });


    //Some checks
    if ((found_lower) && (mview_ == SolverMarketCSRMatrixUpper)) {
        std::cerr << "[Error][SolverMarket][CsrMatrix][read_from_file] mview is upper, but lower elements found\n";
        return MtxReaderErrorUpperViewButLowerFound;
    }
    if ((found_upper) && (mview_ == SolverMarketCSRMatrixLower)) {
        std::cerr << "[Error][SolverMarket][CsrMatrix][read_from_file] mview is lower, but upper elements found\n";
        return MtxReaderErrorLowerViewButUpperFound;
    }
    if (!(found_upper && found_lower) && mview_ == SolverMarketCSRMatrixFull) {
        std::cout << "[Warning][SolverMarket][CsrMatrix][read_from_file] mview is full, but only lower or upper elements found\n";
    }


    // Initialize with 0's
    for (_ITYPE_ i=0; i<n+1; i++){
        offsets_h_(i)=0;
    }

    // First pass: count entries per row
    for (auto& [i, j, val] : entries) {
        if (i >= n || i < 0) {
            std::cerr << "[Error][SolverMarket][CsrMatrix][read_from_file] Invalid row index " << i <<std::endl;
            return MtxReaderErrorOutOfBoundRowIndex;
        }
        if (j >= n || j < 0) {
            std::cerr << "[Error][SolverMarket][CsrMatrix][read_from_file] Invalid col index " << j <<std::endl;
            return MtxReaderErrorOutOfBoundColIndex;
        }
         offsets_h_(i+1)+=1;
    }

    // Prefix sum for empty rows
    for (int i = 0; i < n; ++i) {
        offsets_h_(i + 1) += offsets_h_(i);
    }

    // Second pass: fill in column and value arrays
    std::vector<int> row_fill(n, 0);
    for (auto& [i, j, val] : entries) {
        int offset = offsets_h_(i) + row_fill[i];
        columns_h_(offset) = j;
        values_h_(offset) = val;
        row_fill[i]++;
    }

    // // Detect empty rows
    for (int i = 0; i < n; ++i) {
        if (offsets_h_(i) == offsets_h_(i+1)) {
            std::cout << "[Warning][SolverMarket][CsrMatrix][read_from_file] Row " << i << " is empty\n";
        }
    }

    std::cout << "[Info][SolverMarket][CsrMatrix][read_from_file] Read completed with " << nnz << " nonzeros\n";
    return 0;
}