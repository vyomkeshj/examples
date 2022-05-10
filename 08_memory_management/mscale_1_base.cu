#include <cstdio>
#include <algorithm>



#define CUDACHECK(err) { cuda_check((err), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. '%s' line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        exit(error_code);
    }
}



void print_matrix(float * data, int n_rows, int n_cols, const char * label)
{
    int print_max_size = 8;
    int print_rows = std::min(n_rows, print_max_size);
    int print_cols = std::min(n_cols, print_max_size);

    printf("%s:\n", label);
    for(int row = 0; row < print_rows; row++)
    {
        for(int col = 0; col < print_cols; col++)
        {
            float value = data[row * n_cols + col];
            printf("%7.1f ", value);
        }
        printf("\n");
    }
}

void init_matrix(float * matrix, int n_rows, int n_cols)
{
    for(int row = 0; row < n_rows; row++)
    {
        for(int col = 0; col < n_cols; col++)
        {
            matrix[row * n_cols + col] = 10 * row + col;
        }
    }
}

void check_result(float * matrix, int n_rows, int n_cols, float scalar)
{
    int errorCount = 0;
    for(int row = 0; row < n_rows; row++)
    {
        for(int col = 0; col < n_cols; col++)
        {
            float result = matrix[row * n_cols + col];
            float correct = scalar * (10 * row + col);
            if(result != correct)
            {
                errorCount++;
                if(errorCount <= 5)
                    printf("Incorrect result row %d col %d: correct is %7.1f, but result is %7.1f\n", row, col, correct, result);
            }
        }
    }
    if(errorCount == 0)
    {
        printf("The result is CORRECT!\n");
    }
}



// TODO: matrix scale kernel
__global__ void scale_matrix()


int main()
{
    float scalar = 10.0f;
    int n_rows = 642;
    int n_cols = 531;
    int total_elements = n_rows * n_cols;

    float * h_matrix;
    // TODO: allocate
    init_matrix(h_matrix, n_rows, n_cols);
    print_matrix(h_matrix, n_rows, n_cols, "Input");

    float * d_matrix;
    
    // TODO ...

    print_matrix(h_matrix, n_rows, n_cols, "Output");

    check_result(h_matrix, n_rows, n_cols, scalar);

    // TODO

    return 0;
}
