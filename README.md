# Building Trilinos with CUDA and MueLu

## 1. Load Required Modules

```bash
module load cuda/12.4.0
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
  -D BUILD_SHARED_LIBS:BOOL=ON \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
  -D Trilinos_ENABLE_TESTS:BOOL=OFF \
  -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
  -D Trilinos_ENABLE_MueLu:BOOL=ON \
  -D MueLu_ENABLE_TESTS:BOOL=ON \
  -D MueLu_ENABLE_EXAMPLES:BOOL=ON \
  -D TPL_ENABLE_BLAS:BOOL=ON \
  -D TPL_ENABLE_MPI:BOOL=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ENABLE_DEBUG=ON \
  -D Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
  -D TPL_ENABLE_CUSOLVER=ON \
  -D TPL_ENABLE_CUBLAS=ON \
  ..

make -j 40 && make install
```

---

## 4. Configure Your Solver Project

From the root of your SolverMarket project:

```bash
mkdir build
cd build
cmake ..
```
