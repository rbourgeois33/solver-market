#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

#include "solver-market-header.hpp"
#pragma once


template <typename _TYPE_, typename _ITYPE_=size_t>
class SolverMarketVector {
public:

  SolverMarketVector() = default;

  SolverMarketVector(std::string filename){
    read_matrix_market_file(filename);
  }

  SolverMarketVector(_ITYPE_ n){
    allocate(n);
  }


  SolverMarketVector(_ITYPE_ n, _TYPE_ value){
    allocate(n);
    for (int i=0; i<n; i++){
      values_h_(i)=value;
    }
  }
  
  int read_matrix_market_file(std::string filename);

  int send_to_device();
  _TYPE_* get_host_values_pointer(){return values_h_.data();}
  _TYPE_* get_device_values_pointer(){return values_d_.data();}

  // for a vector, n and nnz are the same thing (we keep 0's)
  _ITYPE_ get_n(){return n_;};
  _ITYPE_ get_nnz(){return n_;};
  _ITYPE_ size(){return n_;};


#ifdef GTEST_ /* only avail for testing. We shouldnt see kokkos outside of the class*/
// Host Views
HostView<_TYPE_> get_host_values()         { return values_h_;  }

// Device Views
DeviceView<_TYPE_> get_device_values()     { return values_d_;  }
#endif

// --- Query Functions for View ---
private:

  _ITYPE_ n_; 
  bool is_allocated_ = false;

  HostView<_TYPE_> values_h_;
  DeviceView<_TYPE_> values_d_;

  int allocate(const _ITYPE_ n);

};
#include "solver-market-vector.tpp"