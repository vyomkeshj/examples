#include <cstdio>
#include <algorithm>



void print_vector(float * data, int count, const char * label)
{
    int print_max = 20;
    int print_count = std::min(count, print_max);

    printf("%s:\n", label);
    for(int i = 0; i < print_count; i++)
        printf("%7.3f ", data[i]);
    printf("\n");
}







int main()
{
    int count = 12345;

    

    return 0;
}
