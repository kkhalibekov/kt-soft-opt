#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdint>

#include <immintrin.h>
#include <omp.h>

#include <chrono>

void writeArrayToFile(const std::string &filename, const uint8_t *array, size_t rows, size_t cols);
uint8_t *readImageFromFile(const std::string &filename, size_t &rows, size_t &cols);

void erosion_naive(const uint8_t *src, uint8_t *dst, int W, int H, int stride, int wx, int wy);
void erosion_naive_simd_omp(const uint8_t *src, uint8_t *dst, int W, int H, int stride, int wx, int wy);

void vhgw_1d_build_min(const uint8_t *src, uint8_t *fw, uint8_t *bw, int length, int window_size);
uint8_t vhgw_1d_get_min(uint8_t *fw, uint8_t *bw, int start, int window_size);
void erosion_vhgw(const uint8_t *src, uint8_t *dst, int W, int H, int wx, int wy);
void erosion_vhgw_simd_omp(const uint8_t *src, uint8_t *dst, int W, int H, int wx, int wy);

int main()
{
    size_t rows, cols;

    uint8_t *img = readImageFromFile("/home/kurbubu/repos/mipt/kt-software-optimization/data/test1_small_grey.txt", rows, cols);

    int W = cols;
    int H = rows;
    int stride = W;
    uint8_t *dst = new uint8_t[H * stride];

    int window_size = 3;
    int wx = window_size / 2;
    int wy = window_size / 2;

    auto start = std::chrono::high_resolution_clock::now();
    erosion_naive(img, dst, W, H, stride, wx, wy);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "erosion_naive img1sg_k3 "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    writeArrayToFile("output_test1_small_grey_k3.txt", dst, rows, cols);

    start = std::chrono::high_resolution_clock::now();
    erosion_naive_simd_omp(img, dst, W, H, stride, wx, wy);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "erosion_naive_simd_omp img1sg_k3  "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    writeArrayToFile("output_test1_small_grey_k3_simd_omp.txt", dst, rows, cols);

    start = std::chrono::high_resolution_clock::now();
    erosion_vhgw(img, dst, W, H, wx, wy);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "erosion_vhgw img1sg_k3  "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    writeArrayToFile("output_test1_small_grey_k3_vanherk.txt", dst, rows, cols);

    start = std::chrono::high_resolution_clock::now();
    erosion_vhgw_simd_omp(img, dst, W, H, wx, wy);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "erosion_vhgw_simd_omp img1sg_k3  "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    writeArrayToFile("output_test1_small_grey_k3_vanherk_simd_omp.txt", dst, rows, cols);

    window_size = 5;
    wx = window_size / 2;
    wy = window_size / 2;

    start = std::chrono::high_resolution_clock::now();
    erosion_naive(img, dst, W, H, stride, wx, wy);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "erosion_naive img1sg_k5  "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    writeArrayToFile("output_test1_small_grey_k5.txt", dst, rows, cols);

    start = std::chrono::high_resolution_clock::now();
    erosion_naive_simd_omp(img, dst, W, H, stride, wx, wy);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "erosion_naive_simd_omp img1sg_k5 "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    writeArrayToFile("output_test1_small_grey_k5_simd_omp.txt", dst, rows, cols);

    start = std::chrono::high_resolution_clock::now();
    erosion_vhgw(img, dst, W, H, wx, wy);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "erosion_vhgw img1sg_k5 "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    writeArrayToFile("output_test1_small_grey_k5_vanherk.txt", dst, rows, cols);

    start = std::chrono::high_resolution_clock::now();
    erosion_vhgw_simd_omp(img, dst, W, H, wx, wy);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "erosion_vhgw_simd_omp img1sg_k5  "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    writeArrayToFile("output_test1_small_grey_k5_vanherk_simd_omp.txt", dst, rows, cols);

    window_size = 7;
    wx = window_size / 2;
    wy = window_size / 2;

    start = std::chrono::high_resolution_clock::now();
    erosion_naive(img, dst, W, H, stride, wx, wy);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "erosion_naive img1sg_k7 "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    writeArrayToFile("output_test1_small_grey_k7.txt", dst, rows, cols);

    start = std::chrono::high_resolution_clock::now();
    erosion_naive_simd_omp(img, dst, W, H, stride, wx, wy);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "erosion_naive_simd_omp img1sg_k7 "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    writeArrayToFile("output_test1_small_grey_k7_simd_omp.txt", dst, rows, cols);

    start = std::chrono::high_resolution_clock::now();
    erosion_vhgw(img, dst, W, H, wx, wy);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "erosion_vhgw img1sg_k7 "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    writeArrayToFile("output_test1_small_grey_k7_vanherk.txt", dst, rows, cols);

    start = std::chrono::high_resolution_clock::now();
    erosion_vhgw_simd_omp(img, dst, W, H, wx, wy);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "erosion_vhgw_simd_omp img1sg_k7  "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    writeArrayToFile("output_test1_small_grey_k7_vanherk_simd_omp.txt", dst, rows, cols);

    delete[] img;
    delete[] dst;

    return 0;
}

uint8_t *readImageFromFile(const std::string &filename, size_t &rows, size_t &cols)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Error: Unable to open file " + filename);
    }

    std::vector<uint8_t> temp_data;
    rows = 0;
    cols = 0;
    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        int value;
        size_t current_col_count = 0;
        while (ss >> value)
        {
            temp_data.push_back(static_cast<uint8_t>(value));
            current_col_count++;
        }
        if (cols == 0)
        {
            cols = current_col_count; // Set the number of columns from the first row
        }
        else if (current_col_count != cols)
        {
            throw std::runtime_error("Error: Row " + std::to_string(rows + 1) + " has a different number of columns.");
        }
        rows++;
    }
    file.close();

    // Convert the vector to a single contiguous 1D array
    size_t total_size = rows * cols;
    uint8_t *image_data = new uint8_t[total_size];
    std::copy(temp_data.begin(), temp_data.end(), image_data);

    return image_data;
}

void writeArrayToFile(const std::string &filename, const uint8_t *array, size_t rows, size_t cols)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Error: Unable to open file " + filename);
    }

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            file << static_cast<int>(array[i * cols + j]);
            if (j < cols - 1)
                file << " "; // Separate values with a space
        }
        file << "\n"; // Newline after each row
    }
    file.close();
}

void erosion_naive(const uint8_t *src, uint8_t *dst, int W, int H, int stride, int wx, int wy)
{
    // calculate erosion of grey image
    // with unsigned 8 bit pixel values
    // W * H - image dimensions
    // (2w_x + 1) * (2w_y + 1) - dimensions of the structural element (morphology window)
    // stride - line length considering alignment

    for (int y = 0; y < H; ++y)
    {
        uint8_t *dstLine = dst + stride * y;

        for (int x = 0; x < W; ++x)
        {
            uint8_t minVal = 255;
            // Loop over the morphological window (centered at x, y)

            int v_start = std::max(0, y - wy);
            int v_end = std::min(H - 1, y + wy);
            for (int v = v_start; v <= v_end; ++v)
            {
                const uint8_t *srcLine = src + stride * v;

                int u_start = std::max(0, x - wx);
                int u_end = std::min(W - 1, x + wx);
                for (int u = u_start; u <= u_end; ++u)
                {
                    minVal = std::min(minVal, srcLine[u]);
                }
            }
            dstLine[x] = minVal;
        }
    }
}

void erosion_naive_simd_omp(const uint8_t *src, uint8_t *dst, int W, int H, int stride, int wx, int wy)
{
#pragma omp parallel for
    for (int y = 0; y < H; ++y)
    {
        uint8_t *dstLine = dst + stride * y;

        int v_start = (y - wy < 0) ? 0 : (y - wy);
        int v_end = (y + wy >= H) ? (H - 1) : (y + wy);

        for (int x = 0; x < W; ++x)
        {
            uint8_t minVal = 255;

            int u_start = (x - wx < 0) ? 0 : (x - wx);
            int u_end = (x + wx >= W) ? (W - 1) : (x + wx);

            __m256i min_vec = _mm256_set1_epi8(255);

            for (int v = v_start; v <= v_end; ++v)
            {
                const uint8_t *srcLine = src + stride * v;

                int u = u_start;
                for (; u <= u_end - 31; u += 32)
                {
                    __m256i data = _mm256_loadu_si256((__m256i const *)(srcLine + u));
                    min_vec = _mm256_min_epu8(min_vec, data);
                }

                alignas(32) uint8_t tmp[32];
                _mm256_store_si256((__m256i *)tmp, min_vec);
                for (int i = 0; i < 32; ++i)
                {
                    if (tmp[i] < minVal)
                        minVal = tmp[i];
                }

                for (; u <= u_end; ++u)
                {
                    uint8_t val = srcLine[u];
                    if (val < minVal)
                        minVal = val;
                }

                min_vec = _mm256_set1_epi8(minVal);
            }
            dstLine[x] = minVal;
        }
    }
}

void vhgw_1d_build_min(const uint8_t *src, uint8_t *fw, uint8_t *bw, int length, int window_size)
{
    int ws = window_size - 1;

    // forward
    int cnt = 0;
    for (int i = 0; i < length; i++)
    {
        if (cnt)
        {
            fw[i] = std::min(src[i], fw[i - 1]);
        }
        else
        {
            fw[i] = src[i];
            cnt = ws;
        }
        cnt--;
    }

    // backward
    bw[length - 1] = src[length - 1];
    for (int i = length - 2; i >= 0; i--)
    {
        if ((i + 1) % ws)
        {
            bw[i] = std::min(src[i], bw[i + 1]);
        }
        else
        {
            bw[i] = src[i];
        }
    }
}

uint8_t vhgw_1d_get_min(uint8_t *fw, uint8_t *bw, int center, int radius)
{
    // return std::min(bw[start], fw[start + window_size - 1]);
    return std::min(bw[center - radius], fw[center + radius]);
}

void erosion_vhgw(const uint8_t *src, uint8_t *dst, int W, int H, int wx, int wy)
{
    // padding mode - BORDER_CONSTANT, border_value=255
    uint8_t border_value = 255;

    int Wpad = W + 2 * wx;
    int Hpad = H + 2 * wy;
    uint8_t *src_pad = new uint8_t[Wpad * Hpad];
    for (int y = 0; y < Hpad; y++)
    {
        for (int x = 0; x < Wpad; x++)
        {
            uint8_t val;
            if (x < wx || x >= W + wx ||
                y < wy || y >= H + wy)
            {
                val = border_value;
            }
            else
            {
                val = src[(y - wy) * W + (x - wx)];
            }
            src_pad[y * Wpad + x] = val;
        }
    }

    // horizontal pass
    uint8_t *temp = new uint8_t[W * Hpad];

    int horizontal_ws = 2 * wx + 1;
    uint8_t *fw = new uint8_t[Wpad];
    uint8_t *bw = new uint8_t[Wpad];

    for (int y = wy; y < H + wy; y++)
    {
        const uint8_t *row_in = src_pad + y * Wpad;
        uint8_t *row_out = temp + y * W;

        vhgw_1d_build_min(row_in, fw, bw, Wpad, horizontal_ws);

        for (int x = wx; x < W + wx; x++)
        {
            row_out[x] = vhgw_1d_get_min(fw, bw, x, wx);
        }
    }
    delete[] fw;
    delete[] bw;

    // vertical pass
    int vertical_ws = 2 * wy + 1;
    fw = new uint8_t[Hpad];
    bw = new uint8_t[Hpad];
    uint8_t *col_in = new uint8_t[Hpad];

    for (int x = 0; x < W; x++)
    {
        // Extract the column
        for (int y = 0; y < Hpad; y++)
        {
            col_in[y] = temp[y * W + x];
        }

        vhgw_1d_build_min(col_in, fw, bw, Hpad, vertical_ws);

        for (int y = wy; y < H + wy; y++)
        {
            dst[(y - wy) * W + x] = vhgw_1d_get_min(fw, bw, y, wy);
        }
    }
    delete[] fw;
    delete[] bw;
    delete[] col_in;

    delete[] src_pad;
    delete[] temp;
}

void erosion_vhgw_simd_omp(const uint8_t *src, uint8_t *dst, int W, int H, int wx, int wy)
{
    uint8_t border_value = 255;

    int Wpad = W + 2 * wx;
    int Hpad = H + 2 * wy;
    uint8_t *src_pad = new uint8_t[Wpad * Hpad];

    int simd_width = 256 / 8;

    {
        __m256i vborder = _mm256_set1_epi8(border_value);
        for (int y = 0; y < Hpad; y++)
        {
            for (int x = 0; x < Wpad;)
            {
                if ((x < wx) || (x >= W + wx) || (y < wy) || (y >= H + wy))
                {
                    // Fill borders
                    int rem = std::min(Wpad - x, simd_width);
                    if (rem == simd_width)
                    {
                        _mm256_storeu_si256((__m256i *)(src_pad + y * Wpad + x), vborder);
                        x += simd_width;
                    }
                    else
                    {
                        for (int k = 0; k < rem; k++)
                            src_pad[y * Wpad + x + k] = border_value;
                        x += rem;
                    }
                }
                else
                {
                    int rem = std::min(Wpad - x, simd_width);
                    __m256i vals;
                    if (rem == simd_width)
                    {
                        vals = _mm256_loadu_si256((__m256i const *)(src + (y - wy) * W + (x - wx)));
                        _mm256_storeu_si256((__m256i *)(src_pad + y * Wpad + x), vals);
                        x += simd_width;
                    }
                    else
                    {
                        for (int k = 0; k < rem; k++)
                            src_pad[y * Wpad + x + k] = src[(y - wy) * W + (x - wx) + k];
                        x += rem;
                    }
                }
            }
        }
    }

    // horizontal pass
    uint8_t *temp = new uint8_t[W * Hpad];
    int horizontal_ws = 2 * wx + 1;
    uint8_t *fw = new uint8_t[Wpad];
    uint8_t *bw = new uint8_t[Wpad];
#pragma omp parallel for
    for (int y = wy; y < H + wy; y++)
    {
        const uint8_t *row_in = src_pad + y * Wpad;
        uint8_t *row_out = temp + y * W;

        vhgw_1d_build_min(row_in, fw, bw, Wpad, horizontal_ws);

        for (int x = wx; x < W + wx; x++)
        {
            row_out[x - wx] = vhgw_1d_get_min(fw, bw, x, wx);
        }
    }

    // vertical pass
    int vertical_ws = 2 * wy + 1;
    delete[] fw;
    delete[] bw;
    fw = new uint8_t[Hpad];
    bw = new uint8_t[Hpad];
    uint8_t *col_in = new uint8_t[Hpad];

#pragma omp parallel for
    for (int x = 0; x < W; x++)
    {
        // Extract the column
        for (int y = 0; y < Hpad; y++)
        {
            col_in[y] = temp[y * W + x];
        }

        vhgw_1d_build_min(col_in, fw, bw, Hpad, vertical_ws);

        for (int y = wy; y < H + wy; y++)
        {
            dst[(y - wy) * W + x] = vhgw_1d_get_min(fw, bw, y, wy);
        }
    }

    delete[] fw;
    delete[] bw;
    delete[] col_in;
    delete[] src_pad;
    delete[] temp;
}
