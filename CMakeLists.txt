cmake_minimum_required(VERSION 3.10)
project(onnx_inference)

set(CMAKE_CXX_STANDARD 17)

add_executable(inference main.cpp)

# Rutas absolutas directas (sin variables)
target_include_directories(inference PRIVATE
    /opt/onnxruntime/include
)

target_link_directories(inference PRIVATE
    /opt/onnxruntime/lib
)

target_link_libraries(inference PRIVATE onnxruntime)