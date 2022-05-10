#include <cstdio>
#include <algorithm>

#define TILE_SIZE 512
#define MASK_RADIUS 8


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

void print_vector(float * data, ssize_t count, const char * label)
{
    ssize_t print_max = 20;
    ssize_t print_count = std::min(count, print_max);

    printf("%s:\n", label);
    for(ssize_t i = 0; i < print_count; i++)
        printf("%7.3f ", data[i]);
    printf("\n");
}



__global__ void generate_input(float * input, ssize_t array_size)
{
    ssize_t idx = (ssize_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < array_size)
        input[idx] = idx + 1;
}

__global__ void generate_mask(float * mask, ssize_t mask_radius)
{
    ssize_t idx = (ssize_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(idx <= mask_radius)
    {
        mask[idx] = idx + 1;
    }
    if(idx < mask_radius)
    {    
        mask[mask_radius + idx + 1] = mask_radius - idx;
    }
}

__global__ void check_output(float * output, ssize_t array_size, ssize_t mask_radius, unsigned long long * errors)
{
    ssize_t idx = (ssize_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= array_size)
        return;

    float r = mask_radius;
    float i = idx;
    
    float observed = output[idx];
    float correct = (r + 1) * (r + 1) * (i + 1);
    if(idx < mask_radius)
    {
        float s = mask_radius - idx;
        correct -= idx*s*(s+1)/2 + s*(s+1)*(2*s+1)/6 - mask_radius*s*(s+1)/2;
    }
    if(idx >= array_size - mask_radius)
    {
        float s = idx + mask_radius + 1 - array_size;
        correct -= (idx+mask_radius+2)*s*(s+1)/2 - s*(s+1)*(2*s+1)/6;
    }

    if(std::abs((correct - observed) / correct) > 1e-5)
    {
        unsigned long long e = atomicAdd(errors, 1);
        if(e < 5)
            printf("Incorrect output, idx %lu, correct is %f, but observed %f\n", idx, correct, observed);
    }
}






// TODO: 1D convolution kernel
__global__
void conv1D(float* input, float* mask, float* output, ssize_t array_size, ssize_t mask_radius) {
    ssize_t idx = (ssize_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx >= array_size)
        return;
    
    int j = -1;
    float sum = 0;
    for(ssize_t i = idx-mask_radius; i <= idx+mask_radius; ++i) {
        j++;
        if(i < 0)
            continue;
        if(i >= array_size)
            continue;
        
        sum += mask[j]*input[i];
    }
    output[idx] = sum;
}



// TODO: 1D convolution kernel using shared memory and tiling
__global__
void conv1D_stencil(float* output, float* mask, float* input, ssize_t array_size) {
  
    __shared__ float shared_data[TILE_SIZE+2*MASK_RADIUS];
    ssize_t idx = blockIdx.x * TILE_SIZE + threadIdx.x - MASK_RADIUS;
    
    if(idx < 0 || idx >= array_size)
        shared_data[threadIdx.x] = 0;
    else
        shared_data[threadIdx.x] = input[idx];
    __syncthreads();
    
    if(threadIdx.x < TILE_SIZE)
    {
        int mask_size = 2*MASK_RADIUS+1;
        float result = 0;
        for(int i = 0; i < mask_size; ++i){
            result += shared_data[threadIdx.x+i] * mask[i];
        }
        output[blockIdx.x * TILE_SIZE + threadIdx.x] = result;
    }
}



int main()
{
    ssize_t array_size = 432109876;
    ssize_t mask_radius = MASK_RADIUS;
    ssize_t mask_size = 2 * mask_radius + 1;

    float * d_input;
    float * d_mask;
    float * d_output;
    unsigned long long * d_errors;
    unsigned long long h_errors = 0;
    int tpb, bpg;

    CUDACHECK(cudaMalloc(&d_input, array_size * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_mask, mask_size * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_output, array_size * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_errors, sizeof(unsigned long long)));
    CUDACHECK(cudaMemcpy(d_errors, &h_errors, sizeof(unsigned long long), cudaMemcpyHostToDevice));


    
    tpb = 1024; bpg = (array_size - 1) / tpb + 1;
    generate_input<<< bpg, tpb >>>(d_input, array_size);
    tpb = 256; bpg = (mask_radius - 1) / tpb + 1;
    generate_mask<<< bpg, tpb >>>(d_mask, mask_radius);
    

    // TODO: compute block and grid dimensions, run the convolution kernel
    //tpb = 1024; bpg = (array_size - 1) / tpb + 1;
    //conv1D<<< bpg, tpb >>>(d_input, d_mask, d_output, array_size, mask_radius);
    
    tpb = TILE_SIZE+2*MASK_RADIUS; bpg = (array_size-1)/TILE_SIZE+1;
    conv1D_stencil<<< bpg, tpb >>>(d_output, d_mask, d_input, array_size);
    
    tpb = 1024; bpg = (array_size - 1) / tpb + 1;
    check_output<<< bpg, tpb >>>(d_output, array_size, mask_radius, d_errors);
    

    // // TODO: compute block and grid dimensions, run the better convolution kernel    
    // tpb = 1024; bpg = (array_size - 1) / tpb + 1;
    // check_output<<< bpg, tpb >>>(d_output, array_size, mask_radius, d_errors);




    CUDACHECK(cudaMemcpy(&h_errors, d_errors, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if(h_errors == 0)
        printf("Everything seems OK\n");
    else
        printf("Total errors: %lu\n", h_errors);



    CUDACHECK(cudaFree(d_input));
    CUDACHECK(cudaFree(d_mask));
    CUDACHECK(cudaFree(d_output));
    CUDACHECK(cudaFree(d_errors));

    return 0;
}
