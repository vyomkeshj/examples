#include <cstdio>
#include <cstring>



#define CUDACHECK(err) { cuda_check((err), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        exit(error_code);
    }
}


void generate_image(unsigned char * image, int width, int height, int blur_size)
{
    memset(image, 0, width * height * sizeof(unsigned char));

    for(int r = blur_size; r < height; r += 2*blur_size+1)
    {
        unsigned char * row = image + r * width;
        for(int c = blur_size; c < width; c += 2*blur_size+1)
        {
            row[c] = 255;
        }
    }
}

void check_blurred_image(unsigned char * image, int width, int height, int blur_size)
{
    int pattern_size = 2*blur_size+1;
    int error_count = 0;

    for(int r = 0; r < height; r++)
    {
        unsigned char * row = image + r * width;

        int sy = 2 * blur_size + 1;
        if(r < blur_size) sy -= blur_size - r;
        if(r >= height - blur_size) sy -= blur_size - (height - r - 1);

        for(int c = 0; c < width; c++)
        {
            int sx = 2 * blur_size + 1;
            if(c < blur_size) sx -= blur_size - c;
            if(c >= width - blur_size) sx -= blur_size - (width - c - 1);
            
            unsigned char true_val;
            if(width % pattern_size <= blur_size && c >= (width / pattern_size) * pattern_size   ||   height % pattern_size <= blur_size && r >= (height / pattern_size) * pattern_size)
                true_val = 0;
            else
                true_val = 255 / (sx * sy);
            unsigned char & observed_val = row[c];

            if(true_val != observed_val)
            {
                if(error_count < 5)
                    printf("Incorrect blur result, error at row=%d, col=%d, true_val=%hhu, observed_val=%hhu\n", r, c, true_val, observed_val);
                error_count++;
            }
        }
    }

    if(error_count > 0)
        printf("Total errors: %d\n", error_count);
    else
        printf("Everything seems OK\n");
}





// TODO: image blur kernel





int main()
{
    int blur_size = 2;
    int width = 451;
    int height = 374;
    unsigned char * image_in;
    unsigned char * image_out;

    CUDACHECK(cudaMallocManaged(&image_in, width * height * sizeof(unsigned char)));
    CUDACHECK(cudaMallocManaged(&image_out, width * height * sizeof(unsigned char)));

    generate_image(image_in, width, height, blur_size);

    // TODO: blur the image

    check_blurred_image(image_out, width, height, blur_size);

    CUDACHECK(cudaFree(image_in));
    CUDACHECK(cudaFree(image_out));

    return 0;
}
