#include <cstdio>
#include <algorithm>
#include "timer.h"


#define CUDACHECK(err) { cuda_check((err), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. '%s' line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        exit(error_code);
    }
}

void print_vector(float * data, int vector_length, const char * label)
{
    int print_max = 20;
    int print_vector_length = std::min(vector_length, print_max);

    printf("%s:\n", label);
    for(int i = 0; i < print_vector_length; i++)
        printf("%7.3f ", data[i]);
    printf("\n");
}

void check_result(float *a, float *b, float *c, int vector_length)
{
    int errorvector_length = 0;
    for(int i = 0; i < vector_length; i++)
    {
        if(c[i] != a[i] + b[i])
        {
            errorvector_length++;
            if(errorvector_length <= 5)
            {
                printf("Error on index %d: correct is %f, but result is %f\n", i, a[i] + b[i], c[i]);
            }
        }
    }
    if(errorvector_length == 0)
    {
        printf("The result is CORRECT!\n");
    }
}


__global__
void vecAdd(float *a, float *b, float *c, int start, int end)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x + start;
 
    // Boundary
    if (id < end)
        c[id] = a[id] + b[id];
}


int main()
{
    int vector_length = 1234567;
    size_t size = sizeof(float)*vector_length;

    float* a;
    cudaMallocManaged(&a, size);
    float* b;
    cudaMallocManaged(&b, size);
    float* c;
    cudaMallocManaged(&c, size);

    for(int i = 0; i < vector_length; i++)
        a[i] = i;
    for(int i = 0; i < vector_length; i++)
        b[i] = 10 * i;
    print_vector(a, vector_length, "Input A");
    print_vector(b, vector_length, "Input B");

    int devs = 0;
    cudaGetDeviceCount(&devs);
    TIMER_BEGIN("multigpu test begin");

    cudaDeviceSynchronize();

    int stride = (vector_length / devs);
    int start = 0;
    int end = stride;
    for(int i = 0; i < devs; ++i){
        cudaSetDevice(i);

        if(i == devs-1)
            end = vector_length;
        vecAdd<<<(end-start-1)/32+1,32>>>(a, b, c, start, end);
//        printf("s: %d, e:%d\n", start, end);
        start = end;
        end += stride; 
    }

    for(int i = 0; i < devs; ++i){
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    TIMER_END("multigpu test begin");


    print_vector(c, vector_length, "Output C");
    check_result(a, b, c, vector_length);

    CUDACHECK(cudaFree(a));
    CUDACHECK(cudaFree(b));
    CUDACHECK(cudaFree(c));

    return 0;
}
