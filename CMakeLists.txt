cmake_minimum_required(VERSION 3.9)

project(easyprof CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)

check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(${PROJECT_NAME}_cuda SHARED "src/${PROJECT_NAME}.cpp")
target_include_directories(${PROJECT_NAME}_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME}_cuda ${CMAKE_DL_LIBS} CUDA::cudart)
endif()

check_language(HIP)

if (CMAKE_HIP_COMPILER)
enable_language(HIP)
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
find_package(HIP REQUIRED)

add_library(${PROJECT_NAME}_hip SHARED "src/${PROJECT_NAME}.cpp")
target_include_directories(${PROJECT_NAME}_hip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME}_hip ${CMAKE_DL_LIBS} HIP::hiprt)
endif()

# TODO Python frontend

