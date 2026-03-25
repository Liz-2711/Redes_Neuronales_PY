#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <onnxruntime_cxx_api.h>

std::vector<uint8_t> leer_imagenes_bin(const std::string& ruta) {
    std::ifstream archivo(ruta, std::ios::binary);
    if (!archivo) {
        throw std::runtime_error("No se pudo abrir el archivo de imagenes: " + ruta);
    }
    std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(archivo)),
                                 std::istreambuf_iterator<char>());
    return buffer;
}

std::vector<uint8_t> leer_etiquetas_bin(const std::string& ruta) {
    std::ifstream archivo(ruta, std::ios::binary);
    if (!archivo) {
        throw std::runtime_error("No se pudo abrir el archivo de etiquetas: " + ruta);
    }
    std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(archivo)),
                                 std::istreambuf_iterator<char>());
    return buffer;
}

int main() {
    try {
        std::cout << "-----> ONNX Runtime C++ Inference <-----" << std::endl;

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MNIST_CPP");
        Ort::SessionOptions session_options;
        
        session_options.SetIntraOpNumThreads(1);
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        std::string model_path = "mnist.onnx";
        Ort::Session session(env, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        auto inputTypeInfo = session.GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto input_shape = inputTensorInfo.GetShape();

        auto outputTypeInfo = session.GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        auto output_shape = outputTensorInfo.GetShape();

        std::cout << "Input Shape: [";
        for (size_t i = 0; i < input_shape.size(); i++) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        int batch_size = (input_shape[0] == -1) ? 1 : input_shape[0];
        std::cout << "Batch Size: " << batch_size << std::endl;
        std::cout << "Layers: 2" << std::endl;
        std::cout << "Activation: ReLU, Softmax" << std::endl;
        std::cout << "Epoch: N/A" << std::endl;
        std::cout << "Loss: N/A" << std::endl;

        auto imagenes_data = leer_imagenes_bin("mnist_test_images.bin");
        auto etiquetas_data = leer_etiquetas_bin("mnist_test_labels.bin");

        const int ALTO = 28;
        const int ANCHO = 28;
        const int TAMANO_IMAGEN = ALTO * ANCHO;
        const int NUM_IMAGENES = imagenes_data.size() / TAMANO_IMAGEN;
        const int NUM_CLASSES = 10;

        std::vector<float> input_tensor_values(TAMANO_IMAGEN);
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::string input_name = session.GetInputNameAllocated(0, allocator).get();
        std::string output_name = session.GetOutputNameAllocated(0, allocator).get();
        const char* input_names[] = {input_name.c_str()};
        const char* output_names[] = {output_name.c_str()};

        int imagenes_a_probar = NUM_IMAGENES;
        int aciertos = 0;
        float total_loss = 0.0f;

        for (int i = 0; i < imagenes_a_probar; i++) {
            size_t offset = i * TAMANO_IMAGEN;
            for (int j = 0; j < TAMANO_IMAGEN; j++) {
                input_tensor_values[j] = static_cast<float>(imagenes_data[offset + j]) / 255.0f;
            }

            std::vector<int64_t> tensor_shape = {1, ALTO, ANCHO};
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor_values.data(),
                TAMANO_IMAGEN,
                tensor_shape.data(),
                tensor_shape.size()
            );

            std::vector<Ort::Value> output_tensors = session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );

            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            std::vector<float> output_values(output_data, output_data + NUM_CLASSES);

            int prediccion = std::distance(output_values.begin(),
                                           std::max_element(output_values.begin(), output_values.end()));
            float max_prob = *std::max_element(output_values.begin(), output_values.end());
            total_loss += -std::log(max_prob + 1e-10f);

            int etiqueta_real = static_cast<int>(etiquetas_data[i]);
            if (prediccion == etiqueta_real) aciertos++;
        }

        float accuracy = (static_cast<float>(aciertos) / imagenes_a_probar) * 100.0f;
        float average_loss = total_loss / imagenes_a_probar;

        std::cout << "Input Shape: [";
        for (size_t i = 0; i < input_shape.size(); i++) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Batch Size: " << batch_size << std::endl;
        std::cout << "Layers: 2" << std::endl;
        std::cout << "Activation: ReLU, Softmax" << std::endl;
        std::cout << "Epoch: N/A" << std::endl;
        std::cout << "Loss: " << average_loss << std::endl;
        std::cout << "Accuracy of all model: " << accuracy << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error fatal: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}