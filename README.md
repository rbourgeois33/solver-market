# Building Trilinos with CUDA and MueLu

## 1. Load Required Modules

ada:
```bash
module load cuda/12.4.0
export OMPI_CXX=/volatile/catA/rb263871/Trilinos/packages/kokkos/bin/nvcc_wrapper 
export CXX=/volatile/catA/rb263871/Trilinos/packages/kokkos/bin/nvcc_wrapper 
```
orcus:
```bash
module load cuda/12.8.0
module load gcc/12.3.0
module load cmake/3.28.3 
module load openmpi/gcc_12.3.0/ 
export OMPI_CXX=/home/catA/rb263871/solver-market/external/trilinos/packages/kokkos/bin/nvcc_wrapper 
export CXX=/home/catA/rb263871/solver-market/external/trilinos/packages/kokkos/bin/nvcc_wrapper 

```
---

## 2. Prepare Trilinos Source

```bash
cd external/trilinos

# Cherry-pick a specific commit for compatibility or fixes
git cherry-pick 89f396b83c7b5a90556204a0878f340f7c491545
```
---

## 3. Configure and Build Trilinos

```bash
mkdir build && cd build

cmake \
  -D CMAKE_INSTALL_PREFIX=../install \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D BUILD_SHARED_LIBS:BOOL=ON \
  -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
  -D Trilinos_ENABLE_TESTS:BOOL=OFF \
  -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
  -D Trilinos_ENABLE_MueLu:BOOL=ON \
  -D Kokkos_ENABLE_DEBUG=OFF \
  -D Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
  -D Kokkos_ARCH_HOPPER90=ON \
  -D TPL_ENABLE_BLAS:BOOL=ON \
  -D TPL_ENABLE_MPI:BOOL=ON \
  -D TPL_ENABLE_CUSOLVER=ON \
  -D TPL_ENABLE_CUBLAS=ON \
  -D TPL_ENABLE_CUDA=ON \
  -D TPL_ENABLE_CUSPARSE=ON \
  -D MueLu_ENABLE_TESTS:BOOL=ON \
  -D MueLu_ENABLE_EXAMPLES:BOOL=ON \
  -D MueLu_ENABLE_CUSPARSE=ON \
  ..

make -j 40 && make install
```

## 3. Configure and Build AMGX

```bash
mkdir build && cd build

cmake ..

make -j 40
```


---

## 4. Configure solver-market

From the root of the repo

```bash
mkdir build
cd build
cmake ..
```

g++ -o ascii2binary ascii2binary.cpp 

ascii2binary aij_2592000.mtx aij_2592000.bin

./muelu_input_deck --xml=../src/muelu/params-files/my-chebyshev.xml --matrix=../../matrix-market-TRUST/aij_2592000.bin --binary=1 --timings --stacked-timer --rhs=../../matrix-market-TRUST/rhs_2592000.mtx 
