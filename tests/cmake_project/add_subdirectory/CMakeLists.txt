# The sample C++ Project use NovaPhy by find_package

cmake_minimum_required(VERSION 3.15..4)

project(NovaPhyFindPackageTest LANGUAGES CXX)

# Before build the project, copy NovaPhy source code to ./NovaPhy directory
add_subdirectory(NovaPhy)

# Export the novaphy::core target for use in this project

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE novaphy::core)