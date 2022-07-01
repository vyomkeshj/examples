#include <cstdio>
 
 
__global__ 
void vecScalar(int *c, int scalar, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Boundary check
    if (id < n)
        c[id] = scalar*c[id];
}


int main()
{
    int n = 10;
    int scalar = 3;

    size_t size = sizeof(int)*n;
    int* h_vector = (int*) malloc(size);
    
    // Print
    for (int i = 0; i < n; ++i) {
        h_vector[i] = i;
        printf("%d, ", h_vector[i]);
    }
    printf("\n");
    
    // GPU
    int* g_vector;
    //todo: how much can be allocated?
    cudaMalloc(&g_vector, size);
    cudaMemcpy(g_vector, h_vector, size, cudaMemcpyHostToDevice);
    // Perform the computation
    /// <<<numBlocks, gridsPerBlock>>>
    vecScalar<<<(n-1)/32+1,32>>>(g_vector, scalar, n);

    cudaMemcpy(h_vector, g_vector, size, cudaMemcpyDeviceToHost);

    cudaFree(g_vector);
    cudaDeviceSynchronize();
    
    // Print
    for (int i = 0; i < n; ++i) {
        printf("%d, ", h_vector[i]);
    }
    printf("\n");
    
    delete[] h_vector;
    return 0;
}

