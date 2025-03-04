#include <stdio.h>

#define prinlnloc printf("Line %d\n", __LINE__)

int main(int argc, char const *argv[])
{
    double **dev;
    cudaMallocHost(&dev, 10 * sizeof(double*));
    prinlnloc;
    for (int i = 0; i < 10; i++) {
        cudaMalloc((void**)&dev[i], 10 * sizeof(double));
        prinlnloc;
    }
    
    prinlnloc;


    for (int i = 0; i < 10; i++) {
        cudaFree(dev[i]);
    }
    cudaFreeHost(dev);

    return 0;
}
