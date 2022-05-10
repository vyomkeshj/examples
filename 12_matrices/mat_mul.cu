#include <cstdio>

#define TILE_SIZE 32



#define CUDACHECK(err) { cuda_check((err), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        exit(error_code);
    }
}

#define CUDAMEASURE(command) do { cudaEvent_t b,e; \
    CUDACHECK(cudaEventCreate(&b)); CUDACHECK(cudaEventCreate(&e)); \
    CUDACHECK(cudaEventRecord(b)); command ; CUDACHECK(cudaEventRecord(e)); \
    CUDACHECK(cudaEventSynchronize(e)); \
    float time; CUDACHECK(cudaEventElapsedTime(&time, b, e)); \
    printf("Execution time: %f ms\n", time); \
    CUDACHECK(cudaEventDestroy(b)); CUDACHECK(cudaEventDestroy(e)); } while(false)



__global__ void matrix_init_A(double * matrix, size_t size, size_t ld)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < size && row < size)
        matrix[row * ld + col] = 2 * row + 7 * col;
}

__global__ void matrix_init_B(double * matrix, size_t size, size_t ld)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < size && row < size)
        matrix[row * ld + col] = 5 * row + 3 * col;
}

__global__ void matrix_check_C(double * matrix, size_t size, size_t ld, unsigned long long * errors)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < size && row < size)
    {
        double r = row;
        double c = col;
        double n = size;

        double correct = 6*r*c*n + (10*r+21*c)*n*(n-1)/2 + 35*n*(n-1)*(2*n-1)/6;
        double observed = matrix[row * ld + col];
        if(abs(correct - observed) / correct > 1e-6)
        {
            unsigned long long e = atomicAdd(errors, 1ull);
            if(e < 5)
                printf("Incorrect result at row %lu col %lu: correct is %.0f, but observed %.0f\n", row, col, correct, observed);
        }
    }
}












__global__ void mat_mul_naive(double * A, double * B, double * C, size_t size, size_t ld)
{
    // TODO: basic matrix multiplication kernel
}



__global__ void mat_mul_tiled(double * A, double * B, double * C, size_t size, size_t ld)
{
    // TODO: matrix multiplication kernel using tiling and shared memory
}









int main()
{
    size_t size = 8901;

    double * d_A;
    double * d_B;
    double * d_C;
    unsigned long long * d_errors;
    unsigned long long h_errors;

    size_t pitch;
    CUDACHECK(cudaMallocPitch(&d_A, &pitch, size * sizeof(double), size));
    CUDACHECK(cudaMallocPitch(&d_B, &pitch, size * sizeof(double), size));
    CUDACHECK(cudaMallocPitch(&d_C, &pitch, size * sizeof(double), size));
    size_t ld = pitch / sizeof(double); // equal-sized matrices will have equal pitch and leading dimension
    CUDACHECK(cudaMalloc(&d_errors, sizeof(unsigned long long)));

    dim3 tpb(TILE_SIZE, TILE_SIZE);
    dim3 bpg((size - 1) / tpb.x + 1, (size - 1) / tpb.y + 1);

    matrix_init_A<<< bpg, tpb >>>(d_A, size, ld);
    matrix_init_B<<< bpg, tpb >>>(d_B, size, ld);





    CUDACHECK(cudaMemset(d_errors, 0, sizeof(unsigned long long)));
    CUDACHECK(cudaMemset(d_C, 0, size * pitch));

    CUDAMEASURE((mat_mul_naive<<< bpg, tpb >>>(d_A, d_B, d_C, size, ld)));

    matrix_check_C<<< bpg, tpb >>>(d_C, size, ld, d_errors);
    CUDACHECK(cudaMemcpy(&h_errors, d_errors, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if(h_errors == 0) printf("Everything seems OK\n");
    else printf("Total errors: %lld\n", h_errors);





    CUDACHECK(cudaMemset(d_errors, 0, sizeof(unsigned long long)));
    CUDACHECK(cudaMemset(d_C, 0, size * pitch));

    CUDAMEASURE((mat_mul_tiled<<< bpg, tpb >>>(d_A, d_B, d_C, size, ld)));

    matrix_check_C<<< bpg, tpb >>>(d_C, size, ld, d_errors);
    CUDACHECK(cudaMemcpy(&h_errors, d_errors, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if(h_errors == 0) printf("Everything seems OK\n");
    else printf("Total errors: %lld\n", h_errors);







    CUDACHECK(cudaFree(d_A));
    CUDACHECK(cudaFree(d_B));
    CUDACHECK(cudaFree(d_C));
    CUDACHECK(cudaFree(d_errors));

    return 0;
}
