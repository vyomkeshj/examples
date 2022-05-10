#include <cstdio>


__global__ void cuda_hello(){
    int blockID = blockIdx.x * blockDim.x;
    int threadID = blockID + threadIdx.x;
    printf("Hello from thread: %d/%d , %d/%d global: %d/%d \n", 
           threadIdx.x, blockDim.x, blockIdx.x, gridDim.x, threadID, blockDim.x*gridDim.x);
}

int main() {
    cuda_hello<<<2,4>>>();
    
    printf("Kernel launched, waiting for sync\n");
    
    cudaDeviceSynchronize();
    return 0;
}

