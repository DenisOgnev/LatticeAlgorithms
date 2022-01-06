# LatticeAlgorithms
 
## After cloning
* **Install SFML [depencdencies](https://www.sfml-dev.org/tutorials/2.5/compile-with-cmake.php) (Linux)**
* **Install SFML (or use from 3rdparty)**
* **Install OpenCL**
### Get SFML (if not installed)
```
git submodule update --init --recursive
```
### Build & Run
```
mkdir build
cd build
cmake ..
cmake --build . --config RELEASE
main.exe
```
