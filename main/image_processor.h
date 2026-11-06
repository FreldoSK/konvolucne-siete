#pragma once

#include <cstdint>
#include <cstdio>

#include <dirent.h>
#include <png.h>

// Reads PNG images from flash memory and prepares them for the neural network:
// each image is converted to 28x28 grayscale pixels.
class ImageProcessor {
public:
    static constexpr int kSize = 28;
    static constexpr int kPixelCount = kSize * kSize;

    bool begin();  // Mounts the image folder
    bool next();   // Loads the next image, false if none are left

    const char *name() const;                          // Name of the loaded image
    const uint8_t *pixels() const { return pixels_; }  // Pixels 0..255

private:
    bool load();
    bool decode(png_structp png, png_infop info, FILE *file);
    void addRow(int y, int width, int height);

    DIR *folder_ = nullptr;
    char path_[272] = {};
    uint8_t *row_ = nullptr;  // One image row in RGBA format

    float sums_[kPixelCount] = {};      // Weighted brightness sum per output pixel
    uint32_t row_weights_[kSize] = {};  // Sum of weights per output row
    uint8_t pixels_[kPixelCount] = {};
};
