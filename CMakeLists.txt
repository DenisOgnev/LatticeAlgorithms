cmake_minimum_required(VERSION 3.0)
project(LatticeAlgorithms)

option(BUILD_DOCS "" OFF)
option(BUILD_PARALLEL_BB "" ON)
option(BUILD_GMP "" ON)

file(GLOB SRC
     "src/*.cpp"
     # "src/utils.cpp"
     # "src/algorithms.cpp"
)

add_subdirectory(3rdparty/boost_config)
add_subdirectory(3rdparty/boost_multiprecision)

find_package(OpenMP REQUIRED)
include_directories(${PROJECT_NAME} include)

# add_library(${PROJECT_NAME} ${SRC})
add_executable(${PROJECT_NAME} ${SRC})

if (BUILD_PARALLEL_BB)
     target_compile_definitions(${PROJECT_NAME} PRIVATE PARALLEL_BB)
endif(BUILD_PARALLEL_BB)

if (BUILD_GMP)
     target_compile_definitions(${PROJECT_NAME} PRIVATE GMP)
endif(BUILD_GMP)

target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} gmp libgmp)
target_link_libraries(${PROJECT_NAME} Boost::config Boost::multiprecision)

if (BUILD_DOCS)
     add_subdirectory(tex)
endif(BUILD_DOCS)