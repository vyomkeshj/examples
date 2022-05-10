#include <cstdio>
#include <cstdlib>



#define CUDACHECK(err) { cuda_check((err), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        exit(error_code);
    }
}



void check_row_sums(unsigned int * row_sums, size_t n_rows, size_t n_cols)
{
    size_t error_count = 0;
    for(size_t r = 0; r < n_rows; r++)
    {
        int observed = row_sums[r];
        int correct = n_cols * (n_cols - 1) / 2 + 2 * r * n_cols;
        if(observed != correct)
        {
            if(error_count < 5)
                printf("Incorrect sum in row %ld. Correct is %d, but observed %d\n", r, correct, observed);
            error_count++;
        }
    }

    if(error_count == 0)
        printf("Row sum seems OK\n");
    else
        printf("Total errors: %d\n", error_count);
}

void check_col_sums(unsigned int * col_sums, size_t n_rows, size_t n_cols)
{
    size_t error_count = 0;
    for(size_t c = 0; c < n_cols; c++)
    {
        int observed = col_sums[c];
        int correct = 2 * (n_rows * (n_rows - 1) / 2) + c * n_rows;
        if(observed != correct)
        {
            if(error_count < 5)
                printf("Incorrect sum in col %ld. Correct is %d, but observed %d\n", c, correct, observed);
            error_count++;
        }
    }

    if(error_count == 0)
        printf("Col sum seems OK\n");
    else
        printf("Total errors: %d\n", error_count);
}







__global__ void matrix_init(unsigned int * matrix, size_t n_rows, size_t n_cols, size_t ld)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < n_cols && row < n_rows)
        matrix[row * ld + col] = 2 * row + col;
}


// TODO kernely
__global__ 
void add_matrix_row(unsigned int * matrix, unsigned int * rows, size_t size, int ld, int pitch){
    size_t row = (blockIdx.x * blockDim.x + threadIdx.x); 
    
    if(row > size)
        return;
    
    size_t sum = 0; 
    for(size_t i = 0; i < size; ++i) {
        sum += matrix[row*ld+i];
    }
    
    rows[row] = sum;
}


__global__ 
void add_matrix_col(unsigned int * matrix, unsigned int * cols, size_t size, int ld, int pitch){
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(col > size)
        return;
    
    size_t sum = 0; 
    for(size_t i = 0; i < size; ++i) {
        sum += matrix[col + i*ld];
    }
    
    cols[col] = sum;
}




int main()
{
    size_t size = 98765;

    unsigned int * d_matrix;
    size_t d_pitch;
    CUDACHECK(cudaMallocPitch(&d_matrix, &d_pitch, size * sizeof(unsigned int), size));
    size_t d_ld = d_pitch / sizeof(int);
    // d_ld = size;

    unsigned int * d_row_sums;
    CUDACHECK(cudaMalloc(&d_row_sums, size * sizeof(unsigned int)));

    unsigned int * d_col_sums;
    CUDACHECK(cudaMalloc(&d_col_sums, size * sizeof(unsigned int)));

    unsigned int * h_row_sums;
    unsigned int * h_col_sums;
    CUDACHECK(cudaMallocHost(&h_row_sums, size * sizeof(unsigned int)));
    CUDACHECK(cudaMallocHost(&h_col_sums, size * sizeof(unsigned int)));
    
    cudaEvent_t start_init, end_init, start_rows, end_rows, start_cols, end_cols;
    CUDACHECK(cudaEventCreate(&start_init));
    CUDACHECK(cudaEventCreate(&end_init));
    CUDACHECK(cudaEventCreate(&start_rows));
    CUDACHECK(cudaEventCreate(&end_rows));
    CUDACHECK(cudaEventCreate(&start_cols));
    CUDACHECK(cudaEventCreate(&end_cols));

    dim3 tpb, bpg;
    


    tpb = dim3(32, 32);
    bpg = dim3((size - 1) / tpb.x + 1, (size - 1) / tpb.y + 1);
    CUDACHECK(cudaEventRecord(start_init));
    matrix_init<<< bpg, tpb >>>(d_matrix, size, size, d_ld);
    CUDACHECK(cudaEventRecord(end_init));

    tpb = 512;
    bpg = (size - 1) / tpb.x + 1;
    cudaEventRecord(start_rows);
    add_matrix_row<<<bpg, tpb>>>(d_matrix, d_row_sums, size, d_ld, d_pitch);
    cudaEventRecord(end_rows);

    tpb = 512;
    bpg = (size - 1) / tpb.x + 1;
    cudaEventRecord(start_cols);
    add_matrix_col<<<bpg, tpb>>>(d_matrix, d_col_sums, size, d_ld, d_pitch);
    cudaEventRecord(end_cols);


    CUDACHECK(cudaMemcpy(h_row_sums, d_row_sums, size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(h_col_sums, d_col_sums, size * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    check_row_sums(h_row_sums, size, size);
    check_col_sums(h_col_sums, size, size);



    float time_init, time_rows, time_cols;
    
    
    // TODO: find the time
    cudaEventElapsedTime(&time_init, start_init, end_init);
    cudaEventElapsedTime(&time_rows, start_rows, end_rows);
    cudaEventElapsedTime(&time_cols, start_cols, end_cols);
    printf("\n");
    printf("Matrix init time:              %7.3f ms\n", time_init);
    printf("Summation time in each row:    %7.3f ms\n", time_rows);
    printf("Summation time in each column: %7.3f ms\n", time_cols);
    printf("Using coalesced memory accesses was %5.2f times faster\n", time_rows / time_cols);



    CUDACHECK(cudaEventDestroy(start_init));
    CUDACHECK(cudaEventDestroy(end_init));
    CUDACHECK(cudaEventDestroy(start_rows));
    CUDACHECK(cudaEventDestroy(end_rows));
    CUDACHECK(cudaEventDestroy(start_cols));
    CUDACHECK(cudaEventDestroy(end_cols));

    CUDACHECK(cudaFree(d_matrix));
    CUDACHECK(cudaFree(d_row_sums));
    CUDACHECK(cudaFree(d_col_sums));
    CUDACHECK(cudaFreeHost(h_row_sums));
    CUDACHECK(cudaFreeHost(h_col_sums));

    return 0;
}
