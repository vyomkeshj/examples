#include <cstdio>
#include <algorithm>

#define CUDACHECK(err) { cuda_check((err), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. '%s' line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        exit(error_code);
    }
}

void print_vector(float * data, int count, const char * label)
{
    int print_max = 20;
    int print_count = std::min(count, print_max);

    printf("%s:\n", label);
    for(int i = 0; i < print_count; i++)
        printf("%7.3f ", data[i]);
    printf("\n");
}

void check_result(float * vector, int count, float scalar)
{
    int errorCount = 0;
    for(int i = 0; i < count; i++)
    {
        if(vector[i] != scalar * i)
        {
            errorCount++;
            if(errorCount <= 5)
            {
                printf("Error on index %d: correct is %f, but result is %f\n", i, scalar * i, vector[i]);
            }
        }
    }
    if(errorCount == 0)
    {
        printf("The result is CORRECT!\n");
    }
}

__global__ void vector_scale(float scalar, float * data, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < count)
        data[index] = scalar * data[index];
}

int main()
{
    int count = 12345;

    float * h_data = new float[count];
    for(int i = 0; i < count; i++)
        h_data[i] = i;
    print_vector(h_data, count, "Input");

    float * d_data;
    cudaMalloc(&d_data, count * sizeof(float));
    cudaMemcpy(d_data, h_data, count * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (count + threads_per_block - 1) / threads_per_block;
    // int blocks_per_grid = (count - 1) / threads_per_block + 1;           // also possible
    // int blocks_per_grid = std::ceil((double)count / threads_per_block);  // also possible, but not nice
    vector_scale<<< blocks_per_grid, threads_per_block >>>(2, d_data, count);
    
    CUDACHECK(cudaDeviceSynchronize());   // not needed here, cudaMemcpy already includes implicit synchronization
    cudaMemcpy(h_data, d_data, count * sizeof(float), cudaMemcpyDeviceToHost);

    print_vector(h_data, count, "Output");

    check_result(h_data, count, 2.0f);

    cudaFree(d_data);
    delete[] h_data;

    return 0;
}
