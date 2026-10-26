cmake_minimum_required(VERSION 3.10)
project(onnx_inference)

set(CMAKE_CXX_STANDARD 17)

# Carpeta de ONNX Runtime (la que tiene include/ y lib/).
# En Windows se pasa al configurar: cmake -DONNXRUNTIME_DIR=C:/ruta/onnxruntime-win-x64-1.24.4 ..
set(ONNXRUNTIME_DIR "/opt/onnxruntime" CACHE PATH "Carpeta de ONNX Runtime")

add_executable(inference main.cpp)

target_include_directories(inference PRIVATE ${ONNXRUNTIME_DIR}/include)
target_link_directories(inference PRIVATE ${ONNXRUNTIME_DIR}/lib)
target_link_libraries(inference PRIVATE onnxruntime)

# El ejecutable busca mnist.onnx en la carpeta donde se corre: se copia junto al .exe
add_custom_command(TARGET inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/Redes/Trying_ONNNX/models/mnist.onnx $<TARGET_FILE_DIR:inference>)

# En Windows las DLL de ONNX Runtime tienen que estar junto al .exe
# (si no, Windows puede cargar la versión vieja que trae System32)
if(WIN32)
    add_custom_command(TARGET inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_DIR}/lib/onnxruntime.dll
            ${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll
            $<TARGET_FILE_DIR:inference>)
endif()
