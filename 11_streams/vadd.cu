#include <cstdio>
#include <algorithm>
#include <vector>



#define CUDACHECK(err) { cuda_check((err), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. '%s' line %d\n", error_code, cudaGetErrorString(error_code), file, line);
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



void print_vector(float * data, int count, const char * label)
{
    int print_max = 20;
    int print_count = std::min(count, print_max);

    printf("%s:\n", label);
    for(int i = 0; i < print_count; i++)
        printf("%7.3f ", data[i]);
    printf("\n");
}

void check_result(float * a, float * b, float * c, size_t count)
{
    size_t error_count = 0;
    for(size_t i = 0; i < count; i++)
    {
        if(c[i] != a[i] + b[i])
        {
            error_count++;
            if(error_count <= 5)
            {
                printf("Error on index %lu: correct is %f, but result is %f\n", i, a[i] + b[i], c[i]);
            }
        }
    }
    if(error_count == 0)
    {
        printf("The result is CORRECT!\n");
    }
    else
    {
        printf("Total errors: %lu\n", error_count);
    }
}



__global__ void vector_add(float * a, float * b, float * c, size_t count)
{
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(index < count)
        c[index] = a[index] + b[index];
}



int main()
{
    size_t count = 456789012;

    float * h_a;
    float * h_b;
    float * h_c;
    float * d_a;
    float * d_b;
    float * d_c;

    CUDACHECK(cudaMallocHost(&h_a, count * sizeof(float)));
    CUDACHECK(cudaMallocHost(&h_b, count * sizeof(float)));
    CUDACHECK(cudaMallocHost(&h_c, count * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_a, count * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_b, count * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_c, count * sizeof(float)));

    for(size_t i = 0; i < count; i++)
        h_a[i] = i;
    for(size_t i = 0; i < count; i++)
        h_b[i] = 10 * i;
    //print_vector(h_a, count, "Input A");
    //print_vector(h_b, count, "Input B");



    // classic vector add
    {
        cudaMemset(d_a, 0, count * sizeof(float));
        cudaMemset(d_b, 0, count * sizeof(float));
        cudaMemset(d_c, 0, count * sizeof(float));
        std::fill(h_c, h_c + count, 0.0f);

        CUDAMEASURE(({
            CUDACHECK(cudaMemcpy(d_a, h_a, count * sizeof(float), cudaMemcpyHostToDevice));
            CUDACHECK(cudaMemcpy(d_b, h_b, count * sizeof(float), cudaMemcpyHostToDevice));

            int tpb = 256;
            int bpg = (count - 1) / tpb + 1;
            vector_add<<< bpg, tpb >>>(d_a, d_b, d_c, count);

            CUDACHECK(cudaMemcpy(h_c, d_c, count * sizeof(float), cudaMemcpyDeviceToHost));
        }));

        check_result(h_a, h_b, h_c, count);
    }



    // vector add but with asynchronous copies
    {
        cudaMemset(d_a, 0, count * sizeof(float));
        cudaMemset(d_b, 0, count * sizeof(float));
        cudaMemset(d_c, 0, count * sizeof(float));
        std::fill(h_c, h_c + count, 0.0f);

        CUDAMEASURE(({
            CUDACHECK(cudaMemcpyAsync(d_a, h_a, count * sizeof(float), cudaMemcpyHostToDevice));
            CUDACHECK(cudaMemcpyAsync(d_b, h_b, count * sizeof(float), cudaMemcpyHostToDevice));

            int tpb = 256;
            int bpg = (count - 1) / tpb + 1;
            vector_add<<< bpg, tpb >>>(d_a, d_b, d_c, count);

            CUDACHECK(cudaMemcpyAsync(h_c, d_c, count * sizeof(float), cudaMemcpyDeviceToHost));
        }));
        
        cudaDeviceSynchronize();

        check_result(h_a, h_b, h_c, count);
    }



    // vector add with asynchronous copies using a non-null stream
    {
        cudaMemset(d_a, 0, count * sizeof(float));
        cudaMemset(d_b, 0, count * sizeof(float));
        cudaMemset(d_c, 0, count * sizeof(float));
        std::fill(h_c, h_c + count, 0.0f);
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        

        CUDAMEASURE(({
            CUDACHECK(cudaMemcpyAsync(d_a, h_a, count * sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDACHECK(cudaMemcpyAsync(d_b, h_b, count * sizeof(float), cudaMemcpyHostToDevice, stream));

            int tpb = 256;
            int bpg = (count - 1) / tpb + 1;
            vector_add<<< bpg, tpb, 0, stream >>>(d_a, d_b, d_c, count);

            CUDACHECK(cudaMemcpyAsync(h_c, d_c, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }));
        
        cudaDeviceSynchronize();
        
        cudaStreamDestroy(stream);

        check_result(h_a, h_b, h_c, count);
    }

    
    
    // segmented vector add with asynchronous copies
    {
    	cudaMemset(d_a, 0, count * sizeof(float));
        cudaMemset(d_b, 0, count * sizeof(float));
        cudaMemset(d_c, 0, count * sizeof(float));
        std::fill(h_c, h_c + count, 0.0f);
        
        size_t n_segments = 6;
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        

        CUDAMEASURE(({
        		for(int s=0; s < n_segments; s++)
            {
            	size_t s_start = (s*count)/n_segments;
              size_t s_end = ((s+1)*count)/n_segments;
              size_t s_length = s_end - s_start;
            	CUDACHECK(cudaMemcpyAsync(d_a + s_start, h_a + s_start, s_length * sizeof(float), cudaMemcpyHostToDevice, stream));
            	CUDACHECK(cudaMemcpyAsync(d_b + s_start, h_b + s_start, s_length * sizeof(float), cudaMemcpyHostToDevice, stream));

            	int tpb = 256;
            	int bpg = (s_length - 1) / tpb + 1;
            	vector_add<<< bpg, tpb, 0, stream >>>(d_a + s_start, d_b + s_start, d_c + s_start, s_length);

            	CUDACHECK(cudaMemcpyAsync(h_c + s_start, d_c + s_start, s_length * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }
        }));
        
        cudaDeviceSynchronize();
        
        cudaStreamDestroy(stream);

        check_result(h_a, h_b, h_c, count);
    }

    
    
    // pipelined vector add with asynchronous copies using multiple streams
    {
        cudaMemset(d_a, 0, count * sizeof(float));
        cudaMemset(d_b, 0, count * sizeof(float));
        cudaMemset(d_c, 0, count * sizeof(float));
        std::fill(h_c, h_c + count, 0.0f);
        
        size_t n_segments = 50;
        int n_streams = 4;
        std::vector<cudaStream_t> Streams(n_streams);
        for (int i=0;i<n_streams;i++)
        {
        	cudaStreamCreate(&Streams[i]);
        }

        CUDAMEASURE(({
        		for(int s=0; s < n_segments; s++)
            {
              cudaStream_t & stream = Streams[s%n_streams];
              
            	size_t s_start = (s*count)/n_segments;
              size_t s_end = ((s+1)*count)/n_segments;
              size_t s_length = s_end - s_start;
            	CUDACHECK(cudaMemcpyAsync(d_a + s_start, h_a + s_start, s_length * sizeof(float), cudaMemcpyHostToDevice, stream));
            	CUDACHECK(cudaMemcpyAsync(d_b + s_start, h_b + s_start, s_length * sizeof(float), cudaMemcpyHostToDevice, stream));

            	int tpb = 256;
            	int bpg = (s_length - 1) / tpb + 1;
            	vector_add<<< bpg, tpb, 0, stream >>>(d_a + s_start, d_b + s_start, d_c + s_start, s_length);

            	CUDACHECK(cudaMemcpyAsync(h_c + s_start, d_c + s_start, s_length * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }
        }));
        
        cudaDeviceSynchronize();
        
        for (int i=0;i<n_streams;i++)
        {
        	cudaStreamDestroy(Streams[i]);
        }

        check_result(h_a, h_b, h_c, count);
    }






    CUDACHECK(cudaFreeHost(h_a));
    CUDACHECK(cudaFreeHost(h_b));
    CUDACHECK(cudaFreeHost(h_c));
    CUDACHECK(cudaFree(d_a));
    CUDACHECK(cudaFree(d_b));
    CUDACHECK(cudaFree(d_c));

    return 0;
}
