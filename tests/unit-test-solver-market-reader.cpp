
#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <vector>

#define GTEST_
#include "solver-market-csr-matrix.hpp"
#include "solver-market-vector.hpp"


void write_temp_file(const std::string& filename, const std::string& content) {
    std::ofstream out(filename);
    out << content;
    out.close();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  Kokkos::initialize(argc, argv); {
    // All Kokkos-related tests run inside this block
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
  }
}

TEST(SolverMarketCsrMatrixReader, BasicUnsortedMatrix) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "5 5 4\n"
        "3 2 3.2\n"
        "1 1 1.0\n"
        "2 5 2.5\n"
        "5 5 5.5\n";

    std::string filename = "test1.mtx";
    write_temp_file(filename, content);
    auto matrix =  SolverMarketCSRMatrix<float>(filename, SolverMarketCSRMatrixFull);

    auto n = matrix.get_n();
    auto nnz = matrix.get_nnz();
    auto offsets = matrix.get_host_offsets();
    auto cols = matrix.get_host_columns();
    auto values = matrix.get_host_values();

    ASSERT_EQ(n, 5);
    ASSERT_EQ(nnz, 4);

    EXPECT_FLOAT_EQ(offsets(0), 0);
    EXPECT_FLOAT_EQ(offsets(1), 1); 
    EXPECT_FLOAT_EQ(offsets(2), 2);
    EXPECT_FLOAT_EQ(offsets(3), 3);
    EXPECT_FLOAT_EQ(offsets(4), 3);
    EXPECT_FLOAT_EQ(offsets(5), 4); 

    EXPECT_FLOAT_EQ(values(0), 1.0);
    EXPECT_FLOAT_EQ(values(1), 2.5);
    EXPECT_FLOAT_EQ(values(2), 3.2);
    EXPECT_FLOAT_EQ(values(3), 5.5);

    EXPECT_FLOAT_EQ(cols(0), 0);
    EXPECT_FLOAT_EQ(cols(1), 4);
    EXPECT_FLOAT_EQ(cols(2), 1);
    EXPECT_FLOAT_EQ(cols(3), 4);
}

TEST(SolverMarketCsrMatrixReader, BasicUnsortedMatrixWithAComment) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "%%Comment\n"
        "5 5 4\n"
        "3 2 3.2\n"
        "1 1 1.0\n"
        "2 5 2.5\n"
        "5 5 5.5\n";

    std::string filename = "test1.mtx";
    write_temp_file(filename, content);
    auto matrix =  SolverMarketCSRMatrix<float>(filename, SolverMarketCSRMatrixFull);

    auto n = matrix.get_n();
    auto nnz = matrix.get_nnz();
    auto offsets = matrix.get_host_offsets();
    auto cols = matrix.get_host_columns();
    auto values = matrix.get_host_values();

    ASSERT_EQ(n, 5);
    ASSERT_EQ(nnz, 4);

    EXPECT_FLOAT_EQ(offsets(0), 0);
    EXPECT_FLOAT_EQ(offsets(1), 1); 
    EXPECT_FLOAT_EQ(offsets(2), 2);
    EXPECT_FLOAT_EQ(offsets(3), 3);
    EXPECT_FLOAT_EQ(offsets(4), 3);
    EXPECT_FLOAT_EQ(offsets(5), 4); 

    EXPECT_FLOAT_EQ(values(0), 1.0);
    EXPECT_FLOAT_EQ(values(1), 2.5);
    EXPECT_FLOAT_EQ(values(2), 3.2);
    EXPECT_FLOAT_EQ(values(3), 5.5);

    EXPECT_FLOAT_EQ(cols(0), 0);
    EXPECT_FLOAT_EQ(cols(1), 4);
    EXPECT_FLOAT_EQ(cols(2), 1);
    EXPECT_FLOAT_EQ(cols(3), 4);
}

TEST(SolverMarketCsrMatrixReader, DenseMatrixNNZGreaterThanN_CheckAll) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "5 5 15\n"
        "1 1 1.0\n"
        "1 2 1.1\n"
        "1 3 1.2\n"
        "2 1 2.0\n"
        "2 2 2.1\n"
        "2 3 2.2\n"
        "3 3 3.0\n"
        "3 4 3.1\n"
        "3 5 3.2\n"
        "4 1 4.0\n"
        "4 5 4.1\n"
        "5 1 5.0\n"
        "5 2 5.1\n"
        "5 3 5.2\n"
        "5 5 5.3\n";

    std::string filename = "test_dense.mtx";
    write_temp_file(filename, content);

    auto matrix = SolverMarketCSRMatrix<float>(filename, SolverMarketCSRMatrixFull);

    auto n = matrix.get_n();
    auto nnz = matrix.get_nnz();
    auto offsets = matrix.get_host_offsets();
    auto cols = matrix.get_host_columns();
    auto values = matrix.get_host_values();

    ASSERT_EQ(n, 5);
    ASSERT_EQ(nnz, 15);

    // Offsets: row starts
    ASSERT_EQ(offsets(0), 0);   // Row 0 (1-based row 1)
    ASSERT_EQ(offsets(1), 3);   // Row 1
    ASSERT_EQ(offsets(2), 6);   // Row 2
    ASSERT_EQ(offsets(3), 9);   // Row 3
    ASSERT_EQ(offsets(4), 11);  // Row 4
    ASSERT_EQ(offsets(5), 15);  // End of Row 4

    // All columns (0-based)
    std::vector<int> expected_cols = {
        0, 1, 2,    // Row 0
        0, 1, 2,    // Row 1
        2, 3, 4,    // Row 2
        0, 4,       // Row 3
        0, 1, 2, 4  // Row 4
    };

    // All values
    std::vector<float> expected_vals = {
        1.0, 1.1, 1.2,
        2.0, 2.1, 2.2,
        3.0, 3.1, 3.2,
        4.0, 4.1,
        5.0, 5.1, 5.2, 5.3
    };

    for (int i = 0; i < nnz; ++i) {
        EXPECT_EQ(cols(i), expected_cols[i]) << "Mismatch at col[" << i << "]";
        EXPECT_FLOAT_EQ(values(i), expected_vals[i]) << "Mismatch at val[" << i << "]";
    }
}

TEST(SolverMarketCsrMatrixReader, MtxReaderWrongHeaderOrNoHeader) {
    std::string content =
        "5 5 4\n"
        "3 2 3.2\n"
        "1 1 1.0\n"
        "2 5 2.5\n"
        "5 5 5.5\n";

    std::string filename = "test3.mtx";
    write_temp_file(filename, content);
    auto matrix =  SolverMarketCSRMatrix<float>();
    auto result = matrix.read_matrix_market_file(filename, SolverMarketCSRMatrixFull);

    ASSERT_EQ(result, MtxReaderWrongHeaderOrNoHeader); // Should fail

}

TEST(SolverMarketCsrMatrixReader, MtxReaderUnsupportedObject) {
    std::string content =
        "%%MatrixMarket tensor yolo real general\n"
        "3 3 2\n"
        "1 2 1.0\n"
        "3 1 2.0\n";  // lower triangle

    std::string filename = "test3.mtx";
    write_temp_file(filename, content);
    auto matrix =  SolverMarketCSRMatrix<float>();
    auto result = matrix.read_matrix_market_file(filename, SolverMarketCSRMatrixFull);


    ASSERT_EQ(result, MtxReaderUnsupportedObject); // Should fail
}

TEST(SolverMarketCsrMatrixReader, MtxReaderUnsupportedMatrixType) {
    std::string content =
        "%%MatrixMarket matrix coordinate real hermitian\n"
        "3 3 2\n"
        "1 2 1.0\n"
        "3 1 2.0\n";  // lower triangle

    std::string filename = "test3.mtx";
    write_temp_file(filename, content);
    auto matrix =  SolverMarketCSRMatrix<float>();
    auto result = matrix.read_matrix_market_file(filename, SolverMarketCSRMatrixFull);


    ASSERT_EQ(result, MtxReaderUnsupportedMatrixType); // Should fail
}


TEST(SolverMarketCsrMatrixReader, MtxReaderTypeReadIsNotTypeGiven) {
    std::string content =
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "3 3 2\n"
        "1 2 1.0\n"
        "3 1 2.0\n";  // lower triangle

    std::string filename = "test3.mtx";
    write_temp_file(filename, content);
    auto matrix =  SolverMarketCSRMatrix<float>();
    auto result = matrix.read_matrix_market_file(filename, SolverMarketCSRMatrixFull, SolverMarketCSRMatrixGeneral);


    ASSERT_EQ(result, MtxReaderTypeReadIsNotTypeGiven); // Should fail
}


TEST(SolverMarketCsrMatrixReader, InvalidUpperWithLowerEntry) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 2\n"
        "1 2 1.0\n"
        "3 1 2.0\n";  // lower triangle

    std::string filename = "test2.mtx";
    write_temp_file(filename, content);
    auto matrix =  SolverMarketCSRMatrix<float>();
    auto result = matrix.read_matrix_market_file(filename, SolverMarketCSRMatrixUpper);

    ASSERT_EQ(result, MtxReaderErrorUpperViewButLowerFound); // Should fail
}

TEST(MatrixReaderTest, InvalidLowerWithUpperEntry) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 2\n"
        "2 1 1.0\n"
        "1 3 2.0\n"; // upper triangle

    write_temp_file("test3.mtx", content);

    std::string filename = "test2.mtx";
    write_temp_file(filename, content);
    auto matrix =  SolverMarketCSRMatrix<float>();
    auto result = matrix.read_matrix_market_file(filename, SolverMarketCSRMatrixLower);

    ASSERT_EQ(result, MtxReaderErrorLowerViewButUpperFound); // Should fail

}


TEST(MatrixReaderTest, EmptyRowsPresent) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "4 4 2\n"
        "1 1 1.0\n"
        "4 4 4.0\n";

        std::string filename = "test1.mtx";
    write_temp_file(filename, content);
    auto matrix =  SolverMarketCSRMatrix<float>(filename, SolverMarketCSRMatrixFull);

    auto n = matrix.get_n();
    auto nnz = matrix.get_nnz();
    auto offsets = matrix.get_host_offsets();
    auto cols = matrix.get_host_columns();
    auto values = matrix.get_host_values();

    ASSERT_EQ(n, 4);
    ASSERT_EQ(nnz, 2);

    ASSERT_EQ(offsets(0), 0);
    ASSERT_EQ(offsets(1), 1); 
    ASSERT_EQ(offsets(2), 1);
    ASSERT_EQ(offsets(3), 1);
    ASSERT_EQ(offsets(4), 2);

    ASSERT_EQ(values(0), 1.0);
    ASSERT_EQ(values(1), 4.0);

    ASSERT_EQ(cols(0), 0);
    ASSERT_EQ(cols(1), 3);
}


TEST(MatrixReaderTest, FileNotFound) {

    auto matrix =  SolverMarketCSRMatrix<float>();
    std::string filename = "idontexist.mtx";
    auto result = matrix.read_matrix_market_file(filename, SolverMarketCSRMatrixUpper);

    ASSERT_EQ(result,  MtxReaderErrorFileNotFound); // Should fail
}

TEST(MatrixReaderTest, SortedOutputCSR) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "4 4 5\n"
        "3 2 3.0\n"
        "1 1 1.0\n"
        "4 4 4.0\n"
        "2 3 2.0\n"
        "2 2 1.5\n";  // deliberately out-of-order input

    std::string filename = "test1.mtx";

    write_temp_file(filename, content);

    auto matrix =  SolverMarketCSRMatrix<float>(filename, SolverMarketCSRMatrixFull);

    auto n = matrix.get_n();
    auto nnz = matrix.get_nnz();
    auto offsets = matrix.get_host_offsets();
    auto cols = matrix.get_host_columns();
    auto values = matrix.get_host_values();

    ASSERT_EQ(n, 4);
    ASSERT_EQ(nnz, 5);

    ASSERT_EQ(offsets(0), 0);
    ASSERT_EQ(offsets(1), 1);  // Row 0 has 1 entry
    ASSERT_EQ(offsets(2), 3);  // Row 1 has 2 entries
    ASSERT_EQ(offsets(3), 4);  // Row 2 has 1 entry
    ASSERT_EQ(offsets(4), 5);  // Row 3 has 1 entry

    ASSERT_EQ(cols(0), 0);     // Row 0
    ASSERT_EQ(cols(1), 1);     // Row 1
    ASSERT_EQ(cols(2), 2);     // Row 1
    ASSERT_EQ(cols(3), 1);     // Row 2
    ASSERT_EQ(cols(4), 3);     // Row 3

    ASSERT_DOUBLE_EQ(values(0), 1.0);
    ASSERT_DOUBLE_EQ(values(1), 1.5);
    ASSERT_DOUBLE_EQ(values(2), 2.0);
    ASSERT_DOUBLE_EQ(values(3), 3.0);
    ASSERT_DOUBLE_EQ(values(4), 4.0);
}


TEST(MatrixReaderTest, InvalidRowIndex) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 2\n"
        "-12 1 1.0\n"
        "3 2 2.0\n";  // only lower

    std::string filename = "test1.mtx";

    write_temp_file(filename, content);

    auto matrix =  SolverMarketCSRMatrix<float>();
    auto result = matrix.read_matrix_market_file(filename, SolverMarketCSRMatrixFull);

    ASSERT_EQ(result, MtxReaderErrorOutOfBoundRowIndex); // Should fail

}

TEST(MatrixReaderTest, InvalidColIndex) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 3\n"
        "1 1 1.0\n"
        "3 -2 2.0\n"
        "2 3 2.0\n";  // only lower

    std::string filename = "test1.mtx";

    write_temp_file(filename, content);

    auto matrix =  SolverMarketCSRMatrix<float>();
    auto result = matrix.read_matrix_market_file(filename, SolverMarketCSRMatrixFull);

    ASSERT_EQ(result, MtxReaderErrorOutOfBoundColIndex); // Should fail
}


TEST(MatrixReaderTest, WrongNnz) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 2\n"
        "2 1 1.0\n"
        "3 2 2.0\n"   
        "3 3 2.0\n";  // only lower

    std::string filename = "test1.mtx";

    write_temp_file(filename, content);

    auto matrix =  SolverMarketCSRMatrix<float>();
    auto result = matrix.read_matrix_market_file(filename, SolverMarketCSRMatrixFull);

    ASSERT_EQ(result, MtxReaderErrorWrongNnz); // Should fail
}

TEST(MatrixReaderTest, UnsortedColumnsWithEmptyRows) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "6 6 5\n"
        "6 2 6.0\n" // Row 5: one entry at col 1
        "2 6 2.0\n"  // Row 1: one entry at col 5
        // Row 2: empty
        "4 3 4.0\n"  // Row 3: entry at col 2
        "4 1 3.0\n"  // Row 3: entry at col 0 (unordered)
        // Row 4: empty
        "1 3 1.0\n";  // Row 0: one entry at col 2

     std::string filename = "test1.mtx";

    write_temp_file(filename, content);

    auto matrix =  SolverMarketCSRMatrix<float>();
    auto result = matrix.read_matrix_market_file(filename, SolverMarketCSRMatrixFull);
    auto n = matrix.get_n();
    auto nnz = matrix.get_nnz();
    auto offsets = matrix.get_host_offsets();
    auto cols = matrix.get_host_columns();
    auto values = matrix.get_host_values();
    
    ASSERT_EQ(result, 0);
    ASSERT_EQ(n, 6);
    ASSERT_EQ(nnz, 5);

    // offsets should be size n+1 = 7
    ASSERT_EQ(offsets(0), 0); // row 0
    ASSERT_EQ(offsets(1), 1); // row 1
    ASSERT_EQ(offsets(2), 2); // row 2 (empty)
    ASSERT_EQ(offsets(3), 2); // row 3 (start of data for row 3)
    ASSERT_EQ(offsets(4), 4); // row 4 (empty)
    ASSERT_EQ(offsets(5), 4); // row 5
    ASSERT_EQ(offsets[6], 5); // end

    // Values in original order (unsorted within rows)
    ASSERT_EQ(values(0), 1.0); // row 0, col 2
    ASSERT_EQ(values(1), 2.0); // row 1, col 5
    ASSERT_EQ(values(2), 3.0); // row 3, col 2
    ASSERT_EQ(values(3), 4.0); // row 3, col 0
    ASSERT_EQ(values(4), 6.0); // row 5, col 1

    ASSERT_EQ(cols(0), 2);
    ASSERT_EQ(cols(1), 5);
    ASSERT_EQ(cols(2), 0);
    ASSERT_EQ(cols(3), 2);
    ASSERT_EQ(cols(4), 1);
} 

TEST(SolverMarketVectorReader, BasicVectorRead) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "4 1 4\n"
        "1 1 1.0\n"
        "2 1 2.0\n"
        "3 1 3.0\n"
        "4 1 4.0\n";

    std::string filename = "vector_valid.mtx";
    write_temp_file(filename, content);

    SolverMarketVector<float> vec;
    int result = vec.read_matrix_market_file(filename);

    ASSERT_EQ(result, MtxReaderSuccess);
    ASSERT_EQ(vec.get_n(), 4);
    auto values = vec.get_host_values();
    EXPECT_FLOAT_EQ(values(0), 1.0);
    EXPECT_FLOAT_EQ(values(1), 2.0);
    EXPECT_FLOAT_EQ(values(2), 3.0);
    EXPECT_FLOAT_EQ(values(3), 4.0);
}

TEST(SolverMarketVectorReader, BasicVectorReadWithAComment) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "%% Comment\n"
        "4 1 4\n"
        "1 1 1.0\n"
        "2 1 2.0\n"
        "3 1 3.0\n"
        "4 1 4.0\n";

    std::string filename = "vector_valid.mtx";
    write_temp_file(filename, content);

    SolverMarketVector<float> vec;
    int result = vec.read_matrix_market_file(filename);

    ASSERT_EQ(result, MtxReaderSuccess);
    ASSERT_EQ(vec.get_n(), 4);
    auto values = vec.get_host_values();
    EXPECT_FLOAT_EQ(values(0), 1.0);
    EXPECT_FLOAT_EQ(values(1), 2.0);
    EXPECT_FLOAT_EQ(values(2), 3.0);
    EXPECT_FLOAT_EQ(values(3), 4.0);
}


TEST(SolverMarketVectorReader, MtxReaderErrorFileNotFound) {

    std::string filename = "doesnotexist.mtx";

    SolverMarketVector<float> vec;
    int result = vec.read_matrix_market_file(filename);

    ASSERT_EQ(result, MtxReaderErrorFileNotFound);
}

TEST(SolverMarketVectorReader, MtxReaderUnsupportedObject0) {
    std::string content =
        "%%MatrixMarket tensor yolo real general\n"
        "3 1 3\n"
        "1 2 1.0\n"
        "3 1 2.0\n";  // lower triangle

    std::string filename = "test3.mtx";
    write_temp_file(filename, content);
    SolverMarketVector<float> vec;
    int result = vec.read_matrix_market_file(filename);

    ASSERT_EQ(result, MtxReaderUnsupportedObject); // Should fail
}

TEST(SolverMarketVectorReader, MtxReaderUnsupportedObject1) {
    std::string content =
        "%%MatrixMarket matrix coordinate real hermitian\n"
        "4 1 4\n"
        "1 1 1.0\n"
        "2 1 2.0\n"
        "3 1 3.0\n"
        "4 1 4.0\n";

    std::string filename = "vector_hermitian.mtx";
    write_temp_file(filename, content);

    SolverMarketVector<float> vec;
    int result = vec.read_matrix_market_file(filename);

    ASSERT_EQ(result, MtxReaderUnsupportedMatrixType);
}

TEST(SolverMarketVectorReader, MtxReaderNotAVector0) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "4 2 4\n"
        "1 1 1.0\n"
        "2 1 2.0\n"
        "3 1 3.0\n"
        "4 1 4.0\n"
        "4 2 4.0\n";

    std::string filename = "oe.mtx";
    write_temp_file(filename, content);

    SolverMarketVector<float> vec;
    int result = vec.read_matrix_market_file(filename);

    ASSERT_EQ(result, MtxReaderNotAVector);
}

TEST(SolverMarketVectorReader, MtxReaderNotAVector1) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "4 1 5\n"
        "1 1 1.0\n"
        "2 1 2.0\n"
        "3 1 3.0\n"
        "4 1 4.0\n";

    std::string filename = "oe.mtx";
    write_temp_file(filename, content);

    SolverMarketVector<float> vec;
    int result = vec.read_matrix_market_file(filename);

    ASSERT_EQ(result, MtxReaderNotAVector);
}



TEST(SolverMarketVectorReader, MtxReaderErrorOutOfBoundRowIndex) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "4 1 4\n"
        "1 1 1.0\n"
        "2 1 2.0\n"
        "18 1 3.0\n"
        "4 1 4.0\n";

    std::string filename = "oe.mtx";
    write_temp_file(filename, content);

    SolverMarketVector<float> vec;
    int result = vec.read_matrix_market_file(filename);

    ASSERT_EQ(result, MtxReaderErrorOutOfBoundRowIndex);
}

TEST(SolverMarketVectorReader, MtxReaderErrorOutOfBoundColIndex) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "4 1 4\n"
        "1 1 1.0\n"
        "2 1 2.0\n"
        "3 10 3.0\n"
        "4 1 4.0\n";

    std::string filename = "oe.mtx";
    write_temp_file(filename, content);

    SolverMarketVector<float> vec;
    int result = vec.read_matrix_market_file(filename);

    ASSERT_EQ(result, MtxReaderErrorOutOfBoundColIndex);
}

TEST(SolverMarketVectorReader, MtxReaderWrongHeaderOrNoHeader) {
    std::string content =
        "4 1 4\n"
        "1 1 1.0\n"
        "2 1 2.0\n"
        "3 1 3.0\n"
        "4 1 4.0\n";

    std::string filename = "oe.mtx";
    write_temp_file(filename, content);

    SolverMarketVector<float> vec;
    int result = vec.read_matrix_market_file(filename);

    ASSERT_EQ(result, MtxReaderWrongHeaderOrNoHeader);
}
