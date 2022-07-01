#include <cstdio>



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


__global__ void init_array(int * array, size_t count, int bin_count)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    size_t prime = (1ul << 63) - 165;
    if(i < count)
    {
        size_t index = ((i % count) * (prime % count)) % count;
        int val = (int)(i % bin_count);
        array[index] = val;
    }
}

__global__ void check_result(int * bins, int bin_count, int array_size, int * errors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= bin_count)
        return;
    
    int correct = (array_size + bin_count - idx - 1) / bin_count;
    int observed = bins[idx];

    if(observed != correct)
    {
        int e = atomicAdd(errors, 1);
        if(e < 5)
            printf("Incorrect result in bin %d, correct is %d, but observed %d\n", idx, correct, observed);
    }
}

// TODO 1.1: histogram kernel
__global__
void histogram(int* array, size_t array_size, int* bin, int bin_count){
    int idx = blockIdx.x * blockDim.x;
    
    if(idx >= array_size)
        return;
    
    int value = array[idx];
    atomicAdd(bin + value, 1);
    bin[value] += 1;
}


// TODO 1.2: histogram kernel using shared memory



// TODO 1.3: histogram kernel using shared memory, looping through all array elements

int main()
{
    size_t count = 123456789;
    int bin_count = 100;

    int * d_array;
    int * d_bins;
    int * d_errors;
    int h_errors;

    CUDACHECK(cudaMalloc(&d_array, count * sizeof(int)));
    CUDACHECK(cudaMalloc(&d_bins, bin_count * sizeof(int)));
    CUDACHECK(cudaMalloc(&d_errors, sizeof(int)));

    int tpb, bpg;

    tpb = 512; bpg = (count - 1) / tpb + 1;
    init_array<<<bpg, tpb>>>(d_array, count, bin_count);


    CUDACHECK(cudaMemset(d_bins, 0, bin_count * sizeof(int)));

    // TODO 1.1: set block and grid dimensions, launch the histogram kernel
    tpb = 1024; bpg = (count-1)/tpb +1;
    histogram<<<tpb,bpg>>>(d_array, count, d_bins, bin_count);
    
    h_errors = 0;
    CUDACHECK(cudaMemcpy(d_errors, &h_errors, sizeof(int), cudaMemcpyHostToDevice));
    tpb = 32; bpg = (bin_count - 1) / tpb + 1;
    check_result<<< bpg, tpb>>>(d_bins, bin_count, count, d_errors);
    CUDACHECK(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));
    if(h_errors == 0)
        printf("Everything seems OK\n");
    else
        printf("Total errors: %d\n", h_errors);

    
    //CUDACHECK(cudaMemset(d_bins, 0, bin_count * sizeof(int)));
    
    // TODO 1.2: set block and grid dimensions, launch the shmem histogram kernel

    //h_errors = 0;
    //CUDACHECK(cudaMemcpy(d_errors, &h_errors, sizeof(int), cudaMemcpyHostToDevice));
    //tpb = 32; bpg = (bin_count - 1) / tpb + 1;
    //check_result<<< bpg, tpb >>>(d_bins, bin_count, count, d_errors);
    //CUDACHECK(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));
    //if(h_errors == 0)
    //    printf("Everything seems OK\n");
    //else
    //    printf("Total errors: %d\n", h_errors);
    


    
    
    //CUDACHECK(cudaMemset(d_bins, 0, bin_count * sizeof(int)));

    // TODO 1.3: set block and grid dimensions, launch the shmemm looped histogram kernel

    //h_errors = 0;
    //CUDACHECK(cudaMemcpy(d_errors, &h_errors, sizeof(int), cudaMemcpyHostToDevice));
    //tpb = 32; bpg = (bin_count - 1) / tpb + 1;
    //check_result<<< bpg, tpb >>>(d_bins, bin_count, count, d_errors);
    //CUDACHECK(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));
    //if(h_errors == 0)
    //    printf("Everything seems OK\n");
    //else
    //    printf("Total errors: %d\n", h_errors);




    CUDACHECK(cudaFree(d_array));
    CUDACHECK(cudaFree(d_bins));

    return 0;
}
