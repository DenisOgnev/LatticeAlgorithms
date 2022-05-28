# LatticeAlgorithms
 
## Requirements
* G++ compiler
* GMP library installed (optional)
### Build & Run
```
mkdir build
cd build
cmake -D BUILD_DOCS=OFF BUILD_PARALLEL_BB=OFF BUILD_GMP=OFF .. -G "MinGW Makefiles"
cmake --build . --config RELEASE
main.exe
```
## Flags
* BUILD_DOCS - build .tex docs
* BUILD_PARALLEL_BB - build parallel Branch and Bound algorithm
* BUILD_GMP - build library with GMP
