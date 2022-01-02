# LatticeAlgorithms
 
## After cloning
* **Install SFML [depencdencies](https://www.sfml-dev.org/tutorials/2.5/compile-with-cmake.php) (Linux)**
* **Install OpenCL**
* **Install SFML (or use from 3rdparty)**
### Get SFML
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
