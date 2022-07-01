#include <cstdio>


__global__ void cuda_hello(){
    int block_index = blockIdx.x;
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Thread index %d: \n",thread_index);
    printf("block index %d: \n",block_index);

//    printf("Hello from thread: %d/%d , %d/%d global: %d/%d \n",
//           threadIdx.x, blockDim.x, blockIdx.x, gridDim.x, thread_index, blockDim.x*gridDim.x);
}

int main() {
    cuda_hello<<<2,4>>>();
    
    printf("Kernel launched, waiting for sync\n");
    
    cudaDeviceSynchronize();
    return 0;
}

