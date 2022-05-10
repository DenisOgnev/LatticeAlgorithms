cmake_minimum_required(VERSION 3.0)
project(LatticeAlgorithms)

# add_library(${PROJECT_NAME} src/algorithms.cpp)

# target_include_directories(${PROJECT_NAME}
#      PUBLIC include
# )


file(GLOB SRC
     "src/*.cpp"
)

#set(BUILD_SHARED_LIBS FALSE)
add_subdirectory(3rdparty/boost_config)
add_subdirectory(3rdparty/boost_multiprecision)

#find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)
#include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(include)

add_executable(${PROJECT_NAME} ${SRC})

#target_link_libraries(main ${OpenCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} Boost::config Boost::multiprecision)
#target_link_libraries(main sfml-graphics sfml-audio sfml-window sfml-system)

add_subdirectory(tex)

#file(COPY "src/axpy.cl" DESTINATION ${CMAKE_BINARY_DIR})
#file(COPY "src/axpy.cl" DESTINATION ${CMAKE_BINARY_DIR}/Release)
#file(COPY "src/axpy.cl" DESTINATION ${CMAKE_BINARY_DIR}/Debug)