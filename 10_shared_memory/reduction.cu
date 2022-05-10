#include <cstdio>
#include <algorithm>

#define TPB 512



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

void print_vector(unsigned long long * data, size_t count, const char * label)
{
    size_t print_max = 20;
    size_t print_count = std::min(count, print_max);

    printf("%s:\n", label);
    for(size_t i = 0; i < print_count; i++)
        printf("%7.3f ", data[i]);
    printf("\n");
}



__global__ void init_array(unsigned long long * array, size_t count)
{
    size_t idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < count)
        array[idx] = idx;
}


// TODO: write the sum reduction kernel
__global__
void sumReduction(unsigned long long * data, size_t count, unsigned long long * result){
    extern __shared__ unsigned long long shared_data[];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t local_idx = threadIdx.x;

    if(idx < count)
        shared_data[local_idx] = data[idx];
    else
        shared_data[local_idx] = 0;

    for(size_t offset = blockDim.x/2; offset > 0; offset /= 2){
        __syncthreads();
        if(local_idx < offset) {
            shared_data[local_idx] += shared_data[local_idx + offset];
        }
    }

    if(local_idx == 0) {
        atomicAdd(result, shared_data[0]);
    }
}


int main()
{
    size_t count = 3210987654;

    unsigned long long * d_array;
    unsigned long long * d_result;
    unsigned long long h_result = 0;
    int tpb, bpg;

    CUDACHECK(cudaMalloc(&d_array, count * sizeof(unsigned long long)));
    CUDACHECK(cudaMalloc(&d_result, sizeof(unsigned long long)));
    CUDACHECK(cudaMemcpy(d_result, &h_result, sizeof(unsigned long long), cudaMemcpyHostToDevice));

    tpb = 512; bpg = (count - 1) / tpb + 1;
    init_array<<< bpg, tpb >>>(d_array, count);


    // TODO: launch the sum reduction kernel
    sumReduction<<<bpg, tpb, tpb*sizeof(unsigned long long)>>>(d_array, count, d_result);

    CUDACHECK(cudaMemcpy(&h_result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    unsigned long long correct = (unsigned long long)(count - 1) * count / 2;
    printf("Correct result is  %20llu\n", correct);
    printf("Computed result is %20llu\n", h_result);
    printf("The results%s match\n", (correct == h_result) ? "" : " DO NOT");

    CUDACHECK(cudaFree(d_array));
    CUDACHECK(cudaFree(d_result));

    return 0;
}
