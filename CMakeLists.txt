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

find_package(OpenMP REQUIRED)
include_directories(include)

add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} gmp libgmp)
target_link_libraries(${PROJECT_NAME} Boost::config Boost::multiprecision)


# add_subdirectory(tex)
