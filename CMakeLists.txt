cmake_minimum_required(VERSION 3.2)
project(LatticeAlgorithms)

option(BUILD_DOCS "" OFF)
option(BUILD_PARALLEL "" OFF)
option(BUILD_GMP "" OFF)

file(GLOB SRC
     "src/utils.cpp"
     "src/algorithms.cpp"
)

add_subdirectory(3rdparty/boost_config)
add_subdirectory(3rdparty/boost_multiprecision)

find_package(OpenMP REQUIRED)

add_library(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME} PUBLIC include)

if (BUILD_PARALLEL)
     target_compile_definitions(${PROJECT_NAME} PUBLIC PARALLEL)
endif(BUILD_PARALLEL)

if (BUILD_GMP)
     target_compile_definitions(${PROJECT_NAME} PUBLIC GMP)
     target_link_libraries(${PROJECT_NAME} gmp libgmp)
endif(BUILD_GMP)

target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} Boost::config Boost::multiprecision)

if (BUILD_DOCS)
     add_subdirectory(tex)
endif(BUILD_DOCS)