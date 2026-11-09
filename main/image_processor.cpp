#include "image_processor.h"

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <strings.h>

#include "esp_littlefs.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Everything below mirrors to_fashion() from the Python notebook and Pillow's
// Image.convert("L") and Image.resize(LANCZOS), so both give the same pixels.

static constexpr char kFolder[] = "/imgs";
static constexpr int kPad = 2;
static constexpr int kSide = ImageProcessor::kSize - 2 * kPad;  // 24
static constexpr int kBackgroundTolerance = 12;
static constexpr int kPrecisionBits = 32 - 8 - 2;  // Same as Pillow

static bool isPng(const char *file_name)
{
    const char *extension = strrchr(file_name, '.');
    return extension != nullptr && strcasecmp(extension, ".png") == 0;
}

// Pillow's RGB -> L conversion (ITU-R 601-2 luma, fixed point).
static inline uint32_t luma(const uint8_t *rgba)
{
    return (rgba[0] * 19595 + rgba[1] * 38470 + rgba[2] * 7471 + 0x8000) >> 16;
}

static inline uint8_t clip8(int32_t value)
{
    if (value >= (1 << kPrecisionBits << 8)) {
        return 255;
    }
    if (value <= 0) {
        return 0;
    }
    return value >> kPrecisionBits;
}

static double sinc(double x)
{
    if (x == 0.0) {
        return 1.0;
    }
    x = x * M_PI;
    return sin(x) / x;
}

static double lanczos(double x)
{
    return (-3.0 <= x && x < 3.0) ? sinc(x) * sinc(x / 3) : 0.0;
}

static void freeFilter(ImageProcessor::Filter &filter)
{
    free(filter.bounds);
    free(filter.weights);
    filter = {};
}

// Same as precompute_coeffs() + normalize_coeffs_8bpc() in Pillow.
static bool makeFilter(int in_size, int out_size, ImageProcessor::Filter &filter)
{
    const double scale = static_cast<double>(in_size) / out_size;
    const double filter_scale = scale < 1.0 ? 1.0 : scale;
    const double support = 3.0 * filter_scale;
    const double inverse_scale = 1.0 / filter_scale;

    filter.ksize = static_cast<int>(ceil(support)) * 2 + 1;
    filter.bounds = static_cast<int *>(malloc(2 * out_size * sizeof(int)));
    filter.weights = static_cast<int32_t *>(calloc(out_size * filter.ksize, sizeof(int32_t)));
    double *weights = static_cast<double *>(malloc(filter.ksize * sizeof(double)));
    if (filter.bounds == nullptr || filter.weights == nullptr || weights == nullptr) {
        free(weights);
        return false;
    }

    for (int out = 0; out < out_size; out++) {
        const double center = (out + 0.5) * scale;
        int first = static_cast<int>(center - support + 0.5);
        if (first < 0) {
            first = 0;
        }
        int last = static_cast<int>(center + support + 0.5);
        if (last > in_size) {
            last = in_size;
        }
        const int count = last - first;

        double sum = 0.0;
        for (int i = 0; i < count; i++) {
            weights[i] = lanczos((i + first - center + 0.5) * inverse_scale);
            sum += weights[i];
        }
        int32_t *fixed = &filter.weights[out * filter.ksize];
        for (int i = 0; i < count; i++) {
            const double weight = sum != 0.0 ? weights[i] / sum : weights[i];
            const double scaled = weight * (1 << kPrecisionBits);
            fixed[i] = static_cast<int32_t>(weight < 0 ? scaled - 0.5 : scaled + 0.5);
        }
        filter.bounds[2 * out] = first;
        filter.bounds[2 * out + 1] = count;
    }
    free(weights);
    return true;
}

// One horizontal Lanczos pass over a row (Pillow's ImagingResampleHorizontal_8bpc).
static void filterRow(const uint8_t *in, uint8_t *out, int out_size,
                      const ImageProcessor::Filter &filter)
{
    for (int x = 0; x < out_size; x++) {
        const int first = filter.bounds[2 * x];
        const int count = filter.bounds[2 * x + 1];
        const int32_t *weights = &filter.weights[x * filter.ksize];
        int32_t sum = 1 << (kPrecisionBits - 1);
        for (int i = 0; i < count; i++) {
            sum += in[first + i] * weights[i];
        }
        out[x] = clip8(sum);
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
        printf("%s: failed to load the image\n", name());
    }
    return false;
}

const char *ImageProcessor::name() const
{
    return path_ + sizeof(kFolder);  // Skips "/imgs/"
}

// The whole image does not fit in RAM, so it is decoded twice:
// pass 1 finds the object, pass 2 crops and scales it.
bool ImageProcessor::load()
{
    const int64_t start = esp_timer_get_time();
    bool loaded = read(Pass::kScan);
    const int64_t scanned = esp_timer_get_time();
    loaded = loaded && read(Pass::kResample);
    freeBuffers();
    scan_time_us_ = scanned - start;
    resample_time_us_ = esp_timer_get_time() - scanned;
    if (!loaded) {
        return false;
    }

    // Paste the scaled object into the middle of a black image.
    memset(pixels_, 0, sizeof(pixels_));
    const int left = (kSize - out_width_) / 2;
    const int top = (kSize - out_height_) / 2;
    uint8_t max = 0;
    for (int y = 0; y < out_height_; y++) {
        for (int x = 0; x < out_width_; x++) {
            const uint8_t value = scaled_[y * out_width_ + x];
            pixels_[(top + y) * kSize + left + x] = value;
            if (value > max) {
                max = value;
            }
        }
    }

    // Stretch the contrast to the full range (float math like numpy float32).
    if (max > 0) {
        for (int i = 0; i < kPixelCount; i++) {
            pixels_[i] = static_cast<uint8_t>(pixels_[i] / static_cast<float>(max) * 255.0f);
        }
    }
    return true;
}

bool ImageProcessor::read(Pass pass)
{
    FILE *file = fopen(path_, "rb");
    if (file == nullptr) {
        return false;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    bool loaded = info != nullptr && decode(png, info, file, pass);

    png_destroy_read_struct(&png, &info, nullptr);
    fclose(file);
    free(row_);
    row_ = nullptr;
    return loaded;
}

bool ImageProcessor::decode(png_structp png, png_infop info, FILE *file, Pass pass)
{
    // On a PNG error, libpng jumps back here and setjmp returns nonzero.
    if (setjmp(png_jmpbuf(png))) {
        return false;
    }

    png_init_io(png, file);
    png_read_info(png, info);

    // Convert every kind of PNG to 8-bit RGBA (transparency becomes alpha).
    png_set_expand(png);
    png_set_strip_16(png);
    png_set_gray_to_rgb(png);
    png_set_add_alpha(png, 0xFF, PNG_FILLER_AFTER);
    png_read_update_info(png, info);

    const int width = png_get_image_width(png, info);
    const int height = png_get_image_height(png, info);
    width_ = width;
    height_ = height;
    const bool interlaced = png_get_interlace_type(png, info) != PNG_INTERLACE_NONE;
    if (interlaced) {
        return false;
    }

    // Read the image row by row, the whole image would not fit in RAM.
    row_ = static_cast<uint8_t *>(malloc(png_get_rowbytes(png, info)));
    if (row_ == nullptr) {
        return false;
    }

    if (pass == Pass::kScan) {
        transparent_count_ = 0;
        memset(border_histogram_, 0, sizeof(border_histogram_));
        const int empty[4] = {INT_MAX, -1, INT_MAX, -1};
        memcpy(alpha_box_, empty, sizeof(empty));
        for (auto &box : value_box_) {
            memcpy(box, empty, sizeof(empty));
        }
    }

    const int last_row = pass == Pass::kScan ? height - 1 : y1_;
    for (int y = 0; y <= last_row; y++) {
        // Large images take seconds, so let the system run (watchdog).
        if (y % 64 == 0) {
            vTaskDelay(1);
        }
        png_read_row(png, row_, nullptr);
        if (pass == Pass::kScan) {
            scanRow(y, width, height);
        } else if (y >= y0_) {
            resampleRow(y);
        }
    }

    return pass == Pass::kScan ? findObject(width, height) : true;
}

// Pass 1: collects everything needed to find the object without a second read.
void ImageProcessor::scanRow(int y, int width, int height)
{
    for (int x = 0; x < width; x++) {
        const uint8_t *rgba = &row_[x * 4];
        const uint32_t gray = luma(rgba);
        const uint8_t alpha = rgba[3];

        if (alpha < 250) {
            transparent_count_++;
        }
        if (alpha > 32) {
            int *box = alpha_box_;
            if (x < box[0]) box[0] = x;
            if (x > box[1]) box[1] = x;
            if (y < box[2]) box[2] = y;
            box[3] = y;
        }

        // The object's bounding box depends on the background brightness,
        // which is only known at the end, so keep a box for every gray value.
        int *box = value_box_[gray];
        if (x < box[0]) box[0] = x;
        if (x > box[1]) box[1] = x;
        if (y < box[2]) box[2] = y;
        box[3] = y;

        // Border pixels in the same order as numpy:
        // first row, last row, first column, last column.
        if (y == 0) border_histogram_[gray]++;
        if (y == height - 1) border_histogram_[gray]++;
        if (x == 0) border_histogram_[gray]++;
        if (x == width - 1) border_histogram_[gray]++;
    }
}

bool ImageProcessor::findObject(int width, int height)
{
    // 1) The alpha channel is the most reliable mask if it carries something.
    use_alpha_ = transparent_count_ > 0.02 * (static_cast<double>(width) * height);
    invert_ = false;

    int box[4] = {INT_MAX, -1, INT_MAX, -1};
    if (use_alpha_) {
        memcpy(box, alpha_box_, sizeof(box));
    } else {
        // 2) Without alpha: background = median of the border pixels.
        const uint32_t count = 2 * width + 2 * height;
        int low = -1, high = -1;
        uint32_t seen = 0;
        for (int value = 0; value < 256; value++) {
            seen += border_histogram_[value];
            if (low < 0 && seen > (count - 1) / 2) low = value;
            if (high < 0 && seen > count / 2) high = value;
        }
        const int background2 = low + high;  // Twice the median
        invert_ = background2 > 2 * 127;     // Light background -> invert

        for (int value = 0; value < 256; value++) {
            const bool object = invert_
                ? 2 * value < background2 - 2 * kBackgroundTolerance
                : 2 * value > background2 + 2 * kBackgroundTolerance;
            const int *value_box = value_box_[value];
            if (!object || value_box[1] < 0) {
                continue;
            }
            if (value_box[0] < box[0]) box[0] = value_box[0];
            if (value_box[1] > box[1]) box[1] = value_box[1];
            if (value_box[2] < box[2]) box[2] = value_box[2];
            if (value_box[3] > box[3]) box[3] = value_box[3];
        }
    }

    if (box[1] < 0) {
        printf("%s: no object found\n", name());
        return false;
    }

    // 3) Crop to the bounding box of the object.
    x0_ = box[0];
    x1_ = box[1];
    y0_ = box[2];
    y1_ = box[3];
    return prepareResample();
}

bool ImageProcessor::prepareResample()
{
    // 4) Scale to 24 pixels on the longer side, keeping the aspect ratio.
    // nearbyint() rounds halves to even, like Python's round().
    const int width = x1_ - x0_ + 1;
    const int height = y1_ - y0_ + 1;
    const double scale = static_cast<double>(kSide) / (height > width ? height : width);
    out_width_ = static_cast<int>(nearbyint(width * scale));
    out_height_ = static_cast<int>(nearbyint(height * scale));
    if (out_width_ < 1) out_width_ = 1;
    if (out_height_ < 1) out_height_ = 1;

    if (!makeFilter(width, out_width_, horizontal_) ||
        !makeFilter(height, out_height_, vertical_)) {
        return false;
    }

    // Pillow runs the vertical pass first when it shrinks much more vertically.
    // The order matters because the result is rounded to 8 bits in between.
    vertical_first_ = height - out_height_ > 0 &&
                      height - out_height_ > (width - out_width_) * 2;

    // Most output rows that one source row contributes to (rows of accumulators).
    slots_ = 0;
    for (int y = 0; y < height; y++) {
        int active = 0;
        for (int out = 0; out < out_height_; out++) {
            const int first = vertical_.bounds[2 * out];
            active += y >= first && y < first + vertical_.bounds[2 * out + 1];
        }
        if (active > slots_) {
            slots_ = active;
        }
    }

    const int sums_width = vertical_first_ ? width : out_width_;
    const int sums_rows = vertical_first_ ? slots_ : out_height_;
    gray_ = static_cast<uint8_t *>(malloc(width));
    finished_ = static_cast<uint8_t *>(malloc(vertical_first_ ? width : out_width_));
    sums_ = static_cast<int32_t *>(malloc(sums_rows * sums_width * sizeof(int32_t)));
    return gray_ != nullptr && finished_ != nullptr && sums_ != nullptr;
}

// Pass 2: crops one row, converts it to gray and feeds the Lanczos filter.
void ImageProcessor::resampleRow(int y)
{
    const int width = x1_ - x0_ + 1;
    for (int x = 0; x < width; x++) {
        const uint8_t *rgba = &row_[(x0_ + x) * 4];
        const uint32_t gray = luma(rgba);
        if (use_alpha_) {
            // Transparent parts are darkened to black (numpy float32 math).
            gray_[x] = static_cast<uint8_t>(gray * (rgba[3] / 255.0f));
        } else {
            gray_[x] = invert_ ? 255 - gray : gray;
        }
    }

    // Horizontal first: shrink the row now, otherwise after the vertical pass.
    const uint8_t *source = gray_;
    int source_width = width;
    if (!vertical_first_) {
        filterRow(gray_, finished_, out_width_, horizontal_);
        source = finished_;
        source_width = out_width_;
    }

    const int row = y - y0_;
    for (int out = 0; out < out_height_; out++) {
        const int first = vertical_.bounds[2 * out];
        const int count = vertical_.bounds[2 * out + 1];
        if (row < first || row >= first + count) {
            continue;
        }

        int32_t *sums = &sums_[(vertical_first_ ? out % slots_ : out) * source_width];
        if (row == first) {
            memset(sums, 0, source_width * sizeof(int32_t));
        }
        const int32_t weight = vertical_.weights[out * vertical_.ksize + row - first];
        for (int x = 0; x < source_width; x++) {
            sums[x] += source[x] * weight;
        }

        // Last source row of this output row: round it to 8 bits.
        if (row == first + count - 1) {
            uint8_t *scaled = &scaled_[out * out_width_];
            const int32_t half = 1 << (kPrecisionBits - 1);
            if (vertical_first_) {
                for (int x = 0; x < source_width; x++) {
                    finished_[x] = clip8(sums[x] + half);
                }
                filterRow(finished_, scaled, out_width_, horizontal_);
            } else {
                for (int x = 0; x < source_width; x++) {
                    scaled[x] = clip8(sums[x] + half);
                }
            }
        }
    }
}

void ImageProcessor::freeBuffers()
{
    freeFilter(horizontal_);
    freeFilter(vertical_);
    free(gray_);
    free(finished_);
    free(sums_);
    gray_ = nullptr;
    finished_ = nullptr;
    sums_ = nullptr;
}
