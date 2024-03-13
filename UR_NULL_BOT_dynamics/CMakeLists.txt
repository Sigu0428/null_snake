cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_BUILD_TYPE DEBUG)
# set(CMAKE_BUILD_TYPE RELEASE)
set(EXE_NAME ur-dynamic-cpp)
project(${EXE_NAME} VERSION 0.1.0
    LANGUAGES CXX)

include(CTest)
enable_testing()

SET( EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
SET( CMAKE_MODULE_PATH "$ENV{EIGEN3_DIR}" )

find_package( Eigen3 3.3 REQUIRED )
#


set(SOURCE_FILES 
    src/main_10_links.cpp
    src/ur_robot.cpp
    src/ur_robot_constructor.cpp
    lib/ur_robot.h
)

add_executable(${EXE_NAME}
    ${SOURCE_FILES}
)

target_include_directories(${EXE_NAME} PRIVATE
    lib
    ${EIGEN3_INCLUDE_DIR}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_subdirectory(pybind11)
pybind11_add_module(ur_robot SHARED
    ${SOURCE_FILES}
    src/ur_robot_wrapper.cpp
)


target_include_directories(ur_robot PUBLIC
    lib
    ${EIGEN3_INCLUDE_DIR}
    PRIVATE ${PYBIND11_INCLUDE_DIR}  # from project CMakeLists.txt
    PRIVATE ${pybind11_INCLUDE_DIR}  # from pybind11Config
    PRIVATE ${PYTHON_INCLUDE_DIRS}
)
