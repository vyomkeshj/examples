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



__global__ void matrix_init(int * matrix, size_t n_rows, size_t n_cols, size_t ld)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < n_cols && row < n_rows)
        matrix[row * ld + col] = 2 * row + col;
}

__global__ void matrix_check_transpose(int * matrix, size_t n_rows, size_t n_cols, size_t ld, unsigned long long * errors)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < n_cols && row < n_rows)
    {
        int correct = row + 2 * col;
        int observed = matrix[row * ld + col];
        if(observed != correct)
        {
            unsigned long long e = atomicAdd(errors, 1ull);
            if(e < 5)
                printf("Incorrect result. row %5d col %5d. correct is %5d, but observed %5d\n", (int)row, (int)col, correct, observed);
        }
    }
}






__global__ void matrix_copy(int * matrix_in, int * matrix_out, size_t size, size_t ld)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < size && row < size)
    {
        size_t idx = row * ld + col;
        matrix_out[idx] = matrix_in[idx];
    }
}

__global__ void matrix_transpose(int * matrix_in, int * matrix_out, size_t size, size_t ld)
{
    // TODO: basic matrix transposition kernel
}

__global__ void matrix_transpose_shmem(int * matrix_in, int * matrix_out, size_t size, size_t ld)
{
    // TODO: matrix transposition kernel using tiling and shared memory
}

__global__ void matrix_transpose_shmem_trick(int * matrix_in, int * matrix_out, size_t size, size_t ld)
{
    // TODO: matrix transposition kernel using tilind and shared memory, avoiding bank conflicts
}






int main()
{
    size_t size = 65432;

    int * d_matrix_in;
    int * d_matrix_out;
    unsigned long long * d_errors;
    unsigned long long h_errors = 0;

    size_t pitch;
    CUDACHECK(cudaMallocPitch(&d_matrix_in, &pitch, size * sizeof(int), size));
    CUDACHECK(cudaMallocPitch(&d_matrix_out, &pitch, size * sizeof(int), size));
    // equal-size matrices will have equal pitch
    size_t ld = pitch / sizeof(int);

    CUDACHECK(cudaMalloc(&d_errors, sizeof(unsigned long long)));

    dim3 tpb(TILE_SIZE,TILE_SIZE);
    dim3 bpg((size - 1) / tpb.x + 1, (size - 1) / tpb.y + 1);

    matrix_init<<< bpg, tpb >>>(d_matrix_in, size, size, ld);




    printf("Matrix copy:\n");
    CUDACHECK(cudaMemset2D(d_matrix_out, pitch, 0, size * sizeof(int), size));
    CUDAMEASURE((matrix_copy<<< bpg, tpb >>>(d_matrix_in, d_matrix_out, size, ld)));
    printf("\n");





    printf("Matrix transpose naive:\n");
    CUDACHECK(cudaMemset2D(d_matrix_out, pitch, 0, size * sizeof(int), size));

    CUDAMEASURE((matrix_transpose<<< bpg, tpb >>>(d_matrix_in, d_matrix_out, size, ld)));

    h_errors = 0;
    CUDACHECK(cudaMemcpy(d_errors, &h_errors, sizeof(unsigned long long), cudaMemcpyHostToDevice));
    matrix_check_transpose<<< bpg, tpb >>>(d_matrix_out, size, size, ld, d_errors);
    CUDACHECK(cudaMemcpy(&h_errors, d_errors, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if(h_errors == 0) printf("Everything seems OK\n");
    else printf("Total errors: %llu\n", h_errors);
    printf("\n");
    




    printf("Matrix transpose using shared memory:\n");
    CUDACHECK(cudaMemset2D(d_matrix_out, pitch, 0, size * sizeof(int), size));

    CUDAMEASURE((matrix_transpose_shmem<<< bpg, tpb >>>(d_matrix_in, d_matrix_out, size, ld)));

    h_errors = 0;
    CUDACHECK(cudaMemcpy(d_errors, &h_errors, sizeof(unsigned long long), cudaMemcpyHostToDevice));
    matrix_check_transpose<<< bpg, tpb >>>(d_matrix_out, size, size, ld, d_errors);
    CUDACHECK(cudaMemcpy(&h_errors, d_errors, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if(h_errors == 0) printf("Everything seems OK\n");
    else printf("Total errors: %llu\n", h_errors);
    printf("\n");
    




    printf("Matrix transpose using shared memory with a trick to avoid bank conflicts:\n");
    CUDACHECK(cudaMemset2D(d_matrix_out, pitch, 0, size * sizeof(int), size));

    CUDAMEASURE((matrix_transpose_shmem_trick<<< bpg, tpb >>>(d_matrix_in, d_matrix_out, size, ld)));

    h_errors = 0;
    CUDACHECK(cudaMemcpy(d_errors, &h_errors, sizeof(unsigned long long), cudaMemcpyHostToDevice));
    matrix_check_transpose<<< bpg, tpb >>>(d_matrix_out, size, size, ld, d_errors);
    CUDACHECK(cudaMemcpy(&h_errors, d_errors, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if(h_errors == 0) printf("Everything seems OK\n");
    else printf("Total errors: %llu\n", h_errors);
    printf("\n");





    CUDACHECK(cudaFree(d_matrix_in));
    CUDACHECK(cudaFree(d_matrix_out));
    CUDACHECK(cudaFree(d_errors));

    return 0;
}
