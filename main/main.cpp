// Fashion-MNIST classifier for ESP32-C6.
// Put PNG images into main/imgs and run "idf.py flash monitor".

#include <cstdint>
#include <cstdio>

#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_image_format.h"
#include "esp_littlefs.h"
#include "esp_partition.h"
#include "esp_timer.h"
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

constexpr int kBenchmarkRuns = 20;  // Inference-only throughput measurement

static ImageProcessor images;

static float ms(int64_t us) { return us / 1000.0f; }
static float kb(size_t bytes) { return bytes / 1024.0f; }
static float percent(size_t part, size_t whole) { return whole ? 100.0f * part / whole : 0.0f; }

static void printFlashUsage()
{
    printf("\n=== Flash ===\n");

    uint32_t chip_size = 0;
    if (esp_flash_get_size(nullptr, &chip_size) == ESP_OK) {
        printf("Flash chip:          %.0f KB\n", kb(chip_size));
    }

    // The running firmware: partition size vs. the size of the image inside it.
    const esp_partition_t *app = esp_partition_find_first(
        ESP_PARTITION_TYPE_APP, ESP_PARTITION_SUBTYPE_ANY, nullptr);
    if (app != nullptr) {
        esp_partition_pos_t position = {app->address, app->size};
        esp_image_metadata_t metadata = {};
        if (esp_image_get_metadata(&position, &metadata) == ESP_OK) {
            printf("Firmware (%s):   %.1f KB / %.1f KB (%.1f %%)\n", app->label,
                   kb(metadata.image_len), kb(app->size), percent(metadata.image_len, app->size));
        }
    }
    printf("Model (g_model):     %.1f KB\n", kb(g_model_len));

    size_t total = 0, used = 0;
    if (esp_littlefs_info("storage", &total, &used) == ESP_OK) {
        printf("Images (LittleFS):   %.1f KB / %.1f KB (%.1f %%)\n", kb(used), kb(total),
               percent(used, total));
    }
}

static void printRamUsage(const tflite::MicroInterpreter &interpreter)
{
    printf("\n=== RAM ===\n");

    const size_t heap_total = heap_caps_get_total_size(MALLOC_CAP_DEFAULT);
    const size_t heap_free = heap_caps_get_free_size(MALLOC_CAP_DEFAULT);
    const size_t heap_min_free = heap_caps_get_minimum_free_size(MALLOC_CAP_DEFAULT);
    printf("Heap total:          %.1f KB\n", kb(heap_total));
    printf("Heap used now:       %.1f KB (%.1f %%)\n", kb(heap_total - heap_free),
           percent(heap_total - heap_free, heap_total));
    // The lowest free heap since boot = the peak, reached while preprocessing.
    printf("Heap used peak:      %.1f KB (%.1f %%)\n", kb(heap_total - heap_min_free),
           percent(heap_total - heap_min_free, heap_total));
    printf("Largest free block:  %.1f KB\n",
           kb(heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT)));

    const size_t arena_used = interpreter.arena_used_bytes();
    printf("Tensor arena:        %.1f KB / %.1f KB (%.1f %%)\n", kb(arena_used), kb(kArenaSize),
           percent(arena_used, kArenaSize));

    // High water mark = stack that was never used (FreeRTOS counts in bytes on ESP-IDF).
    const size_t stack_size = CONFIG_ESP_MAIN_TASK_STACK_SIZE;
    const size_t stack_unused = uxTaskGetStackHighWaterMark(nullptr);
    printf("Main task stack:     %.1f KB / %.1f KB peak (%.1f %%)\n",
           kb(stack_size - stack_unused), kb(stack_size), percent(stack_size - stack_unused, stack_size));
}

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

    int count = 0;
    int64_t total_preprocess_us = 0, total_inference_us = 0;
    uint64_t total_pixels = 0;
    const int64_t run_start = esp_timer_get_time();

    while (images.next()) {
        const int64_t preprocess_us = images.scanTimeUs() + images.resampleTimeUs();

        // Input: the model reads pixels 0..255 as int8 -128..127
        for (int i = 0; i < ImageProcessor::kPixelCount; i++) {
            input->data.int8[i] = images.pixels()[i] - 128;
        }

        // Run the neural network
        const int64_t inference_start = esp_timer_get_time();
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("%s: classification failed\n", images.name());
            continue;
        }
        const int64_t inference_us = esp_timer_get_time() - inference_start;

        // Output: the class with the highest value wins
        float probabilities[10];
        float sum = 0.0f;
        int best = 0;
        for (int i = 0; i < 10; i++) {
            probabilities[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
            sum += probabilities[i];
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }

        // The int8 softmax does not sum exactly to 1, so normalize it like the notebook does.
        printf("%s -> %s (%.1f %%)\n", images.name(), CLASS_NAMES[best],
               probabilities[best] / sum * 100.0f);
        printf("  %dx%d px | pass 1 (decode + find object) %.1f ms | "
               "pass 2 (decode + crop + scale) %.1f ms | inference %.1f ms | total %.1f ms\n",
               images.width(), images.height(), ms(images.scanTimeUs()),
               ms(images.resampleTimeUs()), ms(inference_us), ms(preprocess_us + inference_us));

        count++;
        total_preprocess_us += preprocess_us;
        total_inference_us += inference_us;
        total_pixels += static_cast<uint64_t>(images.width()) * images.height();
    }
    const int64_t run_us = esp_timer_get_time() - run_start;

    if (count == 0) {
        printf("No images classified\n");
        return;
    }

    // Inference alone, repeated on the last image for a stable number.
    int64_t benchmark_us = 0;
    for (int i = 0; i < kBenchmarkRuns; i++) {
        const int64_t start = esp_timer_get_time();
        interpreter.Invoke();
        benchmark_us += esp_timer_get_time() - start;
        vTaskDelay(1);  // Watchdog, not part of the measurement
    }
    const float inference_avg_us = static_cast<float>(benchmark_us) / kBenchmarkRuns;

    printf("\n=== Latency (average of %d images) ===\n", count);
    printf("Preprocessing:       %.1f ms\n", ms(total_preprocess_us / count));
    printf("Inference:           %.1f ms\n", ms(total_inference_us / count));
    printf("Total:               %.1f ms\n", ms((total_preprocess_us + total_inference_us) / count));

    printf("\n=== Throughput ===\n");
    printf("End to end:          %.3f images/s (%.2f Mpx/s of source pixels)\n",
           count * 1e6f / run_us, total_pixels / static_cast<float>(run_us));
    printf("Inference only:      %.1f inferences/s (%.2f ms each, average of %d runs)\n",
           1e6f / inference_avg_us, inference_avg_us / 1000.0f, kBenchmarkRuns);

    printFlashUsage();
    printRamUsage(interpreter);
}
