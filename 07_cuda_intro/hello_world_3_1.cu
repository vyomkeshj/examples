#include <cstdio>


__global__
void cuda_hello(){
    printf("Hello World!\n");
}

int main() {
    cuda_hello<<<1,1>>>(); 
    cudaDeviceSynchronize();
    return 0;
}

