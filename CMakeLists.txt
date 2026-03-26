cmake_minimum_required(VERSION 3.28)
project(ScoopVectors LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ScoopVectors INTERFACE)
add_library(Scoop::Vectors ALIAS ScoopVectors)

target_compile_features(ScoopVectors INTERFACE cxx_std_23)
target_include_directories(ScoopVectors INTERFACE include)

add_subdirectory(examples/ollama-word-play)
