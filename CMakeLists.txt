cmake_minimum_required(VERSION 3.0)
project(LatticeAlgotirhms)

file(GLOB SRC
     "src/*.cpp"
)

set(BUILD_SHARED_LIBS FALSE)
add_subdirectory(3rdparty/SFML)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(include)

add_executable(main ${SRC})

target_link_libraries(main ${OpenCL_LIBRARIES})
target_link_libraries(main OpenMP::OpenMP_CXX)
target_link_libraries(main sfml-graphics sfml-audio sfml-window sfml-system)
#file(COPY "src/axpy.cl" DESTINATION ${CMAKE_BINARY_DIR})
#file(COPY "src/axpy.cl" DESTINATION ${CMAKE_BINARY_DIR}/Release)
#file(COPY "src/axpy.cl" DESTINATION ${CMAKE_BINARY_DIR}/Debug)