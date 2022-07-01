#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cstdlib>


#define CUDACHECK(err) { cuda_check((err), __FILE__, __LINE__); }

inline void cuda_check(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file,
                line);
        exit(error_code);
    }
}


void generate_image(unsigned char *image, int width, int height) {
    memset(image, 0, width * height * sizeof(unsigned char));

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int color_offset = 3 * (r * width + c);
            image[color_offset + 0] = (unsigned char) (std::rand() % 256);
            image[color_offset + 1] = (unsigned char) (std::rand() % 256);
            image[color_offset + 2] = (unsigned char) (std::rand() % 256);
        }
    }
}

void check_image(unsigned char *in_image, unsigned char *out_image, int width, int height) {
    /* obfuscated so you dont just copy it :) */
    unsigned char *r7hu = in_image, *tkl0 = out_image, a9j8, jh6z;
    int td92 = width, be1k = height
    , s32q = 0, k4qa = 1, wu2r = 2, hlbz = 3, g92u, tgvw, sdg3, ftg9 = 0;
    for (int cc49 = s32q; cc49 < be1k;
         cc49++) {
        for (tgvw = s32q; tgvw < td92; tgvw++) {
            g92u = cc49 * td92 + tgvw, sdg3 = hlbz * g92u;
            jh6z =
                    (unsigned char) (0.21 * r7hu[sdg3 + s32q] + 0.72 * r7hu[sdg3 + k4qa] + 0.07 * r7hu[sdg3 + wu2r]);
            a9j8 = tkl0[g92u];
            if (std::abs(a9j8 - jh6z) > 1) {
                if (ftg9 < 5)
                    printf("Incorrect grayscale"
                           " result, error at row=%d, col""=%d, correct=%hhu, observed=%hhu\n", cc49, tgvw,
                           jh6z, a9j8);
                ftg9++;
            }
        }
    }
    if (ftg9 > s32q)printf("Total errors: %d\n", ftg9);
    else
        printf
                ("Everything seems OK\n");
}

__global__ void to_grayscale(unsigned char * grayImage,
                             unsigned char * rgbImage,
                             int width, int height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col < width && row < height) {
        // get 1D coordinate for the grayscale image
        int grayOffset = row*width + col;
        // one can think of the RGB image having
        // CHANNEL times columns than the gray scale image
        int rgbOffset = grayOffset*3;
        unsigned char r = rgbImage[rgbOffset + 0]; // red value for pix
        unsigned char g = rgbImage[rgbOffset + 1]; // green value for pix
        unsigned char b = rgbImage[rgbOffset + 2]; // blue value for pix
        // perform the rescaling and store it
        // We multiply by floating point constants
        grayImage[grayOffset] = (unsigned char)(0.21f*r + 0.71f*g + 0.07f*b);
    }
}

int main(int argc, char **argv) {
    int size = 300;
    unsigned char *image_in;
    unsigned char *image_out;
    int num_pixels = size * size;

    CUDACHECK(cudaMallocManaged(&image_in, 3 * num_pixels * sizeof(unsigned char)));
    CUDACHECK(cudaMallocManaged(&image_out, num_pixels * sizeof(unsigned char)));

    generate_image(image_in, size, size);

    /// <<<numBlocks, threadsPerBlock>>>
    to_grayscale<<<1, 256>>>(image_in, image_out, size, size);

    check_image(image_in, image_out, size, size);

    CUDACHECK(cudaFree(image_in));
    CUDACHECK(cudaFree(image_out));

    return 0;
}
