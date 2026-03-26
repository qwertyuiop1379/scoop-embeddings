cmake_minimum_required(VERSION 3.28)
project(OllamaWordPlay LANGUAGES CXX)

file(GLOB_RECURSE SOURCES src/*.cpp)

add_executable(ollama-word-play ${SOURCES})

target_link_libraries(ollama-word-play PRIVATE Scoop::Vectors)

configure_file(data/inputs.txt ${CMAKE_CURRENT_BINARY_DIR}/inputs.txt COPYONLY)
