#include <cstdio>
#include <algorithm>



#define CUDACHECK(err) { cuda_check((err), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        exit(error_code);
    }
}



void print_matrix(float * data, size_t n_rows, size_t n_cols, const char * label)
{
    size_t print_max_size = 8;
    size_t print_rows = std::min(n_rows, print_max_size);
    size_t print_cols = std::min(n_cols, print_max_size);

    printf("%s:\n", label);
    for(size_t row = 0; row < print_rows; row++)
    {
        for(size_t col = 0; col < print_cols; col++)
        {
            float value = data[row * n_cols + col];
            printf("%7.1f ", value);
        }
        printf("\n");
    }
}

void init_matrix(float * matrix, size_t n_rows, size_t n_cols)
{
    for(size_t row = 0; row < n_rows; row++)
    {
        for(size_t col = 0; col < n_cols; col++)
        {
            matrix[row * n_cols + col] = 10 * row + col;
        }
    }
}

void check_result(float * matrix, size_t n_rows, size_t n_cols, float scalar)
{
    size_t errorCount = 0;
    for(size_t row = 0; row < n_rows; row++)
    {
        for(size_t col = 0; col < n_cols; col++)
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



int main()
{
    float scalar = 10.0f;
    size_t n_rows = 642;
    size_t n_cols = 531;

    float * h_matrix;
    // TODO: allocate host memory

    init_matrix(h_matrix, n_rows, n_cols);
    print_matrix(h_matrix, n_rows, n_cols, "Input");

    float * d_matrix;

    // TODO: allocate memory, copy the data, launch the kernel, copy back the result

    

    print_matrix(h_matrix, n_rows, n_cols, "Output");

    check_result(h_matrix, n_rows, n_cols, scalar);

    // TODO: cleanup

    return 0;
}
