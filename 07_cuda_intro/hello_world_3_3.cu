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
    int* vector = (int*) malloc(size);
    
    // Print
    for (int i = 0; i < n; ++i) {
        vector[i] = i;
        printf("%d, ", vector[i]);
    }
    printf("\n");
    
    // GPU
    int* g_vector;
    cudaMalloc(&g_vector, size);
    cudaMemcpy(g_vector, vector, size, cudaMemcpyHostToDevice);
    // Perform the computation.
    vecScalar<<<(n-1)/32+1,32>>>(g_vector, scalar, n);

    cudaMemcpy(vector, g_vector, size, cudaMemcpyDeviceToHost);

    cudaFree(g_vector);
    cudaDeviceSynchronize();
    
    // Print
    for (int i = 0; i < n; ++i) {
        printf("%d, ", vector[i]);
    }
    printf("\n");
    
    delete[] vector;
    return 0;
}

