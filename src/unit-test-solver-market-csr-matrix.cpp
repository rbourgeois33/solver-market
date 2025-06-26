
#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <vector>

#define GTEST_
#include "solver-market-csr-matrix.hpp"



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

TEST(MatrixReaderTest, BasicUnsortedMatrix) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "5 5 4\n"
        "3 2 3.2\n"
        "1 1 1.0\n"
        "2 5 2.5\n"
        "5 5 5.5\n";
    write_temp_file("test1.mtx", content);

    std::string filename = "test1.mtx";
    auto matrix =  SolverMarketCSRMatrix<float>(filename);

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

/* TEST(MatrixReaderTest, InvalidUpperWithLowerEntry) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 2\n"
        "1 2 1.0\n"
        "3 1 2.0\n";  // lower triangle

    write_temp_file("test2.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test2.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_UPPER, false, false);

    ASSERT_EQ(result, MtxReaderErrorUpperViewButLowerFound); // Should fail

    free(offsets);
    free(cols);
    free(values);
}

TEST(MatrixReaderTest, InvalidLowerWithUpperEntry) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 2\n"
        "2 1 1.0\n"
        "1 3 2.0\n"; // upper triangle

    write_temp_file("test3.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test3.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_LOWER, false, false);

    ASSERT_EQ(result, MtxReaderErrorLowerViewButUpperFound); // Should fail
    
    free(offsets);
    free(cols);
    free(values);
}


TEST(MatrixReaderTest, EmptyRowsPresent) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "4 4 2\n"
        "1 1 1.0\n"
        "4 4 4.0\n";

    write_temp_file("test5.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test5.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

    ASSERT_EQ(result, 0);
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


    free(offsets);
    free(cols);
    free(values);
}

TEST(MatrixReaderTest, FileNotFound) {
    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("nonexistent_file.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

    ASSERT_EQ(result,  MtxReaderErrorFileNotFound); // Should fail

    free(offsets);
    free(cols);
    free(values);
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

    write_temp_file("test_sorted.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test_sorted.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

    ASSERT_EQ(result, 0);
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

    free(offsets);
    free(cols);
    free(values);
}

TEST(MatrixReaderTest, InvalidRowIndex) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 2\n"
        "-12 1 1.0\n"
        "3 2 2.0\n";  // only lower

    write_temp_file("test6.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test6.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

    ASSERT_EQ(result, MtxReaderErrorOutOfBoundRowIndex); // Should fail

    free(offsets);
    free(cols);
    free(values);
}

TEST(MatrixReaderTest, InvalidColIndex) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 3\n"
        "1 1 1.0\n"
        "3 -2 2.0\n"
        "2 3 2.0\n";  // only lower

    write_temp_file("test7.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test7.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

    ASSERT_EQ(result, MtxReaderErrorOfBoundColIndex); // Should fail

    free(offsets);
    free(cols);
    free(values);
}


TEST(MatrixReaderTest, WrongNnz) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 2\n"
        "2 1 1.0\n"
        "3 2 2.0\n"   
        "3 3 2.0\n";  // only lower

    write_temp_file("test8.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test8.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

    ASSERT_EQ(result, MtxReaderErrorWrongNnz); // Should fail

    free(offsets);
    free(cols);
    free(values);
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

    write_temp_file("test_unsorted_empty.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test_unsorted_empty.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

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

    free(offsets);
    free(cols);
    free(values);
} */