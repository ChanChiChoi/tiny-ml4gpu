#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#include "data.cu"
using namespace std::chrono;

int
main(){

    size_t size = 200*sizeof(float);
    float *mat = (float *)malloc(size);
    for(u32 i=0;i<200;i++)
        mat[i] = i;

//    for(u32 i=0; i<20;i++){
//        printf("val %d %f\n",i,mat[i]);
//    }

    u32 a = 16;
    u32 b = 50;
    if(a<=b)
     printf("hello\n==============");
    float *mat_device = NULL;
    cudaMalloc((void **)mat_device,size);
    cudaMemcpy(mat_device, mat, size, cudaMemcpyHostToDevice);

    size_t size1 = 50*sizeof(float);
    float *mean = (float *)malloc(size1);
    float *mean_device = nullptr;
    cudaMalloc((void **)mean_device,size1);
    cudaMemcpy(mean_device, mean, size1, cudaMemcpyHostToDevice);
    

    auto t0 = high_resolution_clock::now();
    mean_by_rows_cpu(mat_device,mean_device, 4,50);
    cudaDeviceSynchronize();
    auto t1 = high_resolution_clock::now();

    cudaMemcpy(mat, mat_device, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(mean, mean_device, size1, cudaMemcpyDeviceToHost);

//
    for(u32 i=30; i<40;i++){
        printf("val %d %f %f\n",i,mat[i], mean[i]);
    }

    cudaFree(mean_device);
    cudaFree(mat_device);
    printf("take time %d\n",duration_cast<milliseconds>(t1-t0).count());
    free(mat);
    free(mean);
}

