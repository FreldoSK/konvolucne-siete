#pragma once

#include <cstdint>
#include <cstdio>

#include <dirent.h>
#include <png.h>

// Reads PNG images from flash memory and prepares them for the neural network
// exactly like to_fashion() in the Python notebook: the object is cropped,
// scaled to 24x24 (keeping its aspect ratio, Lanczos filter as in Pillow),
// centered on a black 28x28 image and its contrast is stretched to 0..255.
class ImageProcessor {
public:
    static constexpr int kSize = 28;
    static constexpr int kPixelCount = kSize * kSize;

    bool begin();  // Mounts the image folder
    bool next();   // Loads the next image, false if none are left

    const char *name() const;                          // Name of the loaded image
    const uint8_t *pixels() const { return pixels_; }  // Pixels 0..255

    // Measurements of the last loaded image.
    int width() const { return width_; }
    int height() const { return height_; }
    int64_t scanTimeUs() const { return scan_time_us_; }          // Pass 1: decode + find object
    int64_t resampleTimeUs() const { return resample_time_us_; }  // Pass 2: decode + crop + scale

    // Lanczos filter weights for one dimension, same as Pillow computes them.
    struct Filter {
        int ksize = 0;              // Weights per output pixel
        int *bounds = nullptr;      // First source pixel and count, per output pixel
        int32_t *weights = nullptr; // Fixed point, ksize per output pixel
    };

private:
    enum class Pass { kScan, kResample };

    bool load();
    bool read(Pass pass);
    bool decode(png_structp png, png_infop info, FILE *file, Pass pass);
    void scanRow(int y, int width, int height);
    bool findObject(int width, int height);
    bool prepareResample();
    void resampleRow(int y);
    void freeBuffers();

    DIR *folder_ = nullptr;
    char path_[272] = {};
    uint8_t *row_ = nullptr;  // One image row in RGBA format

    // Pass 1: statistics used to find the object.
    uint32_t transparent_count_ = 0;   // Pixels with alpha < 250
    int alpha_box_[4] = {};            // Bounding box of alpha > 32: x0, x1, y0, y1
    int value_box_[256][4] = {};       // Bounding box of each gray value
    uint32_t border_histogram_[256] = {};

    // Pass 2: crop and resample.
    bool use_alpha_ = false;
    bool invert_ = false;
    bool vertical_first_ = false;
    int x0_ = 0, x1_ = 0, y0_ = 0, y1_ = 0;  // Crop box (inclusive)
    int out_width_ = 0, out_height_ = 0;     // Size of the scaled object
    int slots_ = 0;                          // Accumulator rows
    Filter horizontal_, vertical_;
    uint8_t *gray_ = nullptr;      // Cropped gray row
    uint8_t *finished_ = nullptr;  // Vertically filtered row (vertical first)
    int32_t *sums_ = nullptr;      // Accumulators of the vertical filter
    uint8_t scaled_[24 * 24] = {};

    uint8_t pixels_[kPixelCount] = {};

    int width_ = 0, height_ = 0;
    int64_t scan_time_us_ = 0, resample_time_us_ = 0;
};
