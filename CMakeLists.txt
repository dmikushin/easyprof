cmake_minimum_required(VERSION 3.9)

project(easyprof CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(ThirdParty/sample)

include(CheckLanguage)

check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(${PROJECT_NAME}_cuda SHARED "src/${PROJECT_NAME}.cu")
target_include_directories(${PROJECT_NAME}_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME}_cuda ${CMAKE_DL_LIBS})

set(${PROJECT_NAME}_target ${PROJECT_NAME}_cuda)
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

add_library(${PROJECT_NAME}_hip SHARED "src/${PROJECT_NAME}.cu")
set_source_files_properties(src/${PROJECT_NAME}.cu PROPERTIES LANGUAGE HIP)
target_include_directories(${PROJECT_NAME}_hip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME}_hip ${CMAKE_DL_LIBS})

set(${PROJECT_NAME}_target ${PROJECT_NAME}_hip)
endif()

add_custom_target(${PROJECT_NAME}_test
	COMMAND LD_PRELOAD=$<TARGET_FILE:${${PROJECT_NAME}_target}> PROFILE_TIME=1 $<TARGET_FILE:bilinear_gpu_unoptimized> ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/sample/thefox.bmp)
add_dependencies(${PROJECT_NAME}_test bilinear_gpu_unoptimized)

# TODO Python frontend

