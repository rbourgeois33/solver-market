#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

#pragma once

using Device = Kokkos::DefaultExecutionSpace;
using Host = Kokkos::DefaultHostExecutionSpace;

#ifndef GTEST_ //This is skipped for unit tests as it runs on a simple runner
static_assert(!std::is_same<Device, Host>::value, "Error: Device and Host execution spaces must be different. You build Kokkos from trilinos without GPU option !");
#endif

template<typename _TYPE_>
using HostView=Kokkos::View<_TYPE_*, Host>;

template<typename _TYPE_>
using DeviceView=Kokkos::View<_TYPE_*, Device>;

enum SolverMarketCSRMatrixView {
    SolverMarketCSRMatrixViewNone,
    SolverMarketCSRMatrixFull,
    SolverMarketCSRMatrixLower,
    SolverMarketCSRMatrixUpper
};

enum SolverMarketCSRMatrixType {
    SolverMarketCSRMatrixTypeNone,
    SolverMarketCSRMatrixGeneral,
    SolverMarketCSRMatrixSymmetric,
};

template <typename _TYPE_>
class SolverMarketCSRMatrix {
public:

  SolverMarketCSRMatrix() = default;

  // SolverMarketCSRMatrix(const size_t n, const size_t nnz){
  //   allocate(n, nnz);
  // }

  SolverMarketCSRMatrix(std::string filename, SolverMarketCSRMatrixView mview, SolverMarketCSRMatrixType mtype= SolverMarketCSRMatrixTypeNone){
    read_matrix_market_file(filename, mview, mtype);
  }
  
  int read_matrix_market_file(std::string filename, SolverMarketCSRMatrixView mview, SolverMarketCSRMatrixType mtype= SolverMarketCSRMatrixTypeNone);

  int send_to_device();

  size_t* get_host_offsets_pointer(){return offsets_h_.data();}
  size_t* get_host_columns_pointer(){return columns_h_.data();}
  _TYPE_* get_host_values_pointer(){return values_h_.data();}

  size_t* get_device_offsets_pointer(){return offsets_d_.data();}
  size_t* get_device_columns_pointer(){return columns_d_.data();}
  _TYPE_* get_device_values_pointer(){return values_d_.data();}

  size_t get_n(){return n_;};
  size_t get_nnz(){return nnz_;};

#ifdef GTEST_ /* only avail for testing. We shouldnt see kokkos outside of the class*/
// Host Views
HostView<size_t> get_host_offsets()        { return offsets_h_; }
HostView<size_t> get_host_columns()        { return columns_h_; }
HostView<_TYPE_> get_host_values()         { return values_h_;  }

// Device Views
DeviceView<size_t> get_device_offsets()    { return offsets_d_; }
DeviceView<size_t> get_device_columns()    { return columns_d_; }
DeviceView<_TYPE_> get_device_values()     { return values_d_;  }
#endif

// --- Query Functions for View ---
void setView(SolverMarketCSRMatrixView view) { mview_ = view; }
void setType(SolverMarketCSRMatrixType type) { mtype_ = type; }
SolverMarketCSRMatrixView getView() const { return mview_; }
SolverMarketCSRMatrixType getType() const { return mtype_; }
bool isFull() const { return mview_ == SolverMarketCSRMatrixFull; }
bool isLower() const { return mview_ == SolverMarketCSRMatrixLower; }
bool isUpper() const { return mview_ == SolverMarketCSRMatrixUpper; }
bool isGeneral() const { return mtype_ == SolverMarketCSRMatrixGeneral; }
bool isSymmetric() const { return mtype_ == SolverMarketCSRMatrixSymmetric; }
bool hasValidView() const { return mview_ != SolverMarketCSRMatrixViewNone; }
bool hasValidType() const { return mtype_ != SolverMarketCSRMatrixTypeNone; }

private:

  size_t n_; /* size of the matrix (assumed square)*/
  size_t nnz_; /* # of non-zero elements*/
  bool is_allocated_ = false;

  HostView<size_t> offsets_h_, columns_h_;
  HostView<_TYPE_> values_h_;

  DeviceView<size_t> offsets_d_, columns_d_;
  DeviceView<_TYPE_> values_d_;

  SolverMarketCSRMatrixView mview_=SolverMarketCSRMatrixViewNone;
  SolverMarketCSRMatrixType mtype_=SolverMarketCSRMatrixTypeNone;

  int allocate(const size_t n, const size_t nnz);

};
#include "solver-market-csr-matrix.tpp"