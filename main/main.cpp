// Fashion-MNIST classifier for ESP32-C6.
// Put PNG images into main/imgs and run "idf.py flash monitor".

#include <cstdint>
#include <cstdio>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "image_processor.h"
#include "model_data.h"  // g_model[]

static const char *CLASS_NAMES[10] = {
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot"};

constexpr int kArenaSize = 80 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];

static ImageProcessor images;

extern "C" void app_main(void)
{
    // Load the model
    tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    tflite::MicroInterpreter interpreter(tflite::GetModel(g_model), resolver,
                                         tensor_arena, kArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("Failed to load the model\n");
        return;
    }

    if (!images.begin()) {
        printf("Failed to open the images\n");
        return;
    }

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    while (images.next()) {
        // Input: the model reads pixels 0..255 as int8 -128..127
        for (int i = 0; i < ImageProcessor::kPixelCount; i++) {
            input->data.int8[i] = images.pixels()[i] - 128;
        }

        // Run the neural network
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("%s: classification failed\n", images.name());
            continue;
        }

        // Output: the class with the highest value wins
        int best = 0;
        for (int i = 1; i < 10; i++) {
            if (output->data.int8[i] > output->data.int8[best]) {
                best = i;
            }
        }

        float confidence = (output->data.int8[best] - output->params.zero_point) * output->params.scale;
        printf("%s -> %s (%.1f %%)\n", images.name(), CLASS_NAMES[best], confidence * 100.0f);

        // Processing takes a while, so let the system run briefly (watchdog).
        vTaskDelay(1);
    }
}
