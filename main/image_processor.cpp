#include "image_processor.h"

#include <cstdlib>
#include <cstring>

#include <strings.h>

#include "esp_littlefs.h"

static constexpr char kFolder[] = "/imgs";

static bool isPng(const char *file_name)
{
    const char *extension = strrchr(file_name, '.');
    return extension != nullptr && strcasecmp(extension, ".png") == 0;
}

// Triangle filter: splits a source pixel between the two nearest output
// pixels based on distance. Weights are in the range 0..256.
template <typename AddWeight>
static void spread(int position, int length, AddWeight add_weight)
{
    const int scaled = (2 * position + 1) * ImageProcessor::kSize * 128 / length - 128;
    const int first = scaled >> 8;
    const int second_weight = scaled & 255;

    if (first >= 0) {
        add_weight(first, 256 - second_weight);
    }
    if (first + 1 < ImageProcessor::kSize) {
        add_weight(first + 1, second_weight);
    }
}

bool ImageProcessor::begin()
{
    esp_vfs_littlefs_conf_t config = {};
    config.base_path = kFolder;
    config.partition_label = "storage";
    config.read_only = true;

    if (esp_vfs_littlefs_register(&config) != ESP_OK) {
        return false;
    }
    folder_ = opendir(kFolder);
    return folder_ != nullptr;
}

bool ImageProcessor::next()
{
    while (dirent *entry = readdir(folder_)) {
        if (!isPng(entry->d_name)) {
            continue;
        }
        snprintf(path_, sizeof(path_), "%s/%s", kFolder, entry->d_name);
        if (load()) {
            return true;
        }
        printf("%s: obrazok sa nepodarilo nacitat\n", name());
    }
    return false;
}

const char *ImageProcessor::name() const
{
    return path_ + sizeof(kFolder);  // Skips "/imgs/"
}

bool ImageProcessor::load()
{
    FILE *file = fopen(path_, "rb");
    if (file == nullptr) {
        return false;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    bool loaded = info != nullptr && decode(png, info, file);

    png_destroy_read_struct(&png, &info, nullptr);
    fclose(file);
    free(row_);
    row_ = nullptr;
    return loaded;
}

bool ImageProcessor::decode(png_structp png, png_infop info, FILE *file)
{
    // On a PNG error, libpng jumps back here and setjmp returns nonzero.
    if (setjmp(png_jmpbuf(png))) {
        return false;
    }

    png_init_io(png, file);
    png_read_info(png, info);

    // Convert every kind of PNG to 8-bit RGBA.
    png_set_expand(png);
    png_set_strip_16(png);
    png_set_gray_to_rgb(png);
    png_set_add_alpha(png, 0xFF, PNG_FILLER_AFTER);
    png_read_update_info(png, info);

    const int width = png_get_image_width(png, info);
    const int height = png_get_image_height(png, info);
    const bool interlaced = png_get_interlace_type(png, info) != PNG_INTERLACE_NONE;
    if (width < kSize || height < kSize || interlaced) {
        return false;
    }

    // Read the image row by row, the whole image would not fit in RAM.
    row_ = static_cast<uint8_t *>(malloc(png_get_rowbytes(png, info)));
    if (row_ == nullptr) {
        return false;
    }

    memset(sums_, 0, sizeof(sums_));
    memset(row_weights_, 0, sizeof(row_weights_));
    for (int y = 0; y < height; y++) {
        png_read_row(png, row_, nullptr);
        addRow(y, width, height);
    }

    // Each output pixel is a weighted average of source pixels.
    for (int i = 0; i < kPixelCount; i++) {
        pixels_[i] = sums_[i] / row_weights_[i / kSize] + 0.5f;
    }
    return true;
}

void ImageProcessor::addRow(int y, int width, int height)
{
    uint32_t row[kSize] = {};
    uint32_t weights[kSize] = {};

    // Shrink the row to a width of 28 pixels.
    for (int x = 0; x < width; x++) {
        const uint8_t *rgba = &row_[x * 4];
        const uint32_t gray = (299 * rgba[0] + 587 * rgba[1] + 114 * rgba[2]) / 1000;

        // The model was trained on light clothing on a black background,
        // so transparent parts of the image are darkened to black.
        const uint32_t on_black = gray * rgba[3] / 255;

        spread(x, width, [&](int out_x, uint32_t weight) {
            row[out_x] += on_black * weight;
            weights[out_x] += weight;
        });
    }

    // Add the shrunken row to the two nearest output rows.
    spread(y, height, [&](int out_y, uint32_t weight) {
        for (int out_x = 0; out_x < kSize; out_x++) {
            sums_[out_y * kSize + out_x] += weight * (static_cast<float>(row[out_x]) / weights[out_x]);
        }
        row_weights_[out_y] += weight;
    });
}
