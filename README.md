# LatticeAlgorithms
 
## Requirements
* G++ compiler
* GMP library installed (optional)
## Build & Run
```
mkdir build
cd build
cmake -D BUILD_DOCS=OFF BUILD_PARALLEL_BB=OFF BUILD_GMP=OFF .. -G "MinGW Makefiles"
cmake --build . --config RELEASE
```
## Flags
* BUILD_DOCS - build .tex docs
* BUILD_PARALLEL_BB - build parallel Branch and Bound algorithm
* BUILD_GMP - build library with GMP
## Usage with CMake
* Clone/add as git submodule
```
git clone https://github.com/DenisOgnev/LatticeAlgorithms
```
OR
```
git submodule add https://github.com/DenisOgnev/LatticeAlgorithms 3rdparty/LatticeAlgorithms
```
* Update submodules
```
git submodule git submodule update --init --recursive
```
* Add to CMakeLists.txt
```
set(BUILD_GMP ON/OFF CACHE BOOL "Enable GMP"/"Disable GMP")
set(BUILD_PARALLEL_BB ON/OFF CACHE BOOL "Don't build parallel B&B"/"Build parallel B&B")
set(BUILD_DOCS ON/OFF CACHE BOOL "Build docs"/"Don't build docs") # Requires XeLatex comiler and packages

add_subdirectory(libs/LatticeAlgorithms)

target_link_libraries(${PROJECT_NAME} LatticeAlgorithms)
```
* Include in .cpp
```
#include "algorithms.hpp"
```
