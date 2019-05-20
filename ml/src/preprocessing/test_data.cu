#include <stdio.h>
#include <cuda_runtime.h>
//#include <chrono>

#include "data.cu"
#include "common/include/helper.cuh"
//using namespace std::chrono;

int
main(){

    size_t rows = 4;
    size_t cols = 50;
    size_t size = rows*cols*sizeof(float);
    float *mat = (float *)malloc(size);
    for(u32 i=0;i<cols;i++)
        mat[i] = i;

//    for(u32 i=0; i<20;i++){
//        printf("val %d %f\n",i,mat[i]);
//    }

    float *mat_device;
    cudaMalloc((void **)&mat_device,size);
    cudaMemcpy(mat_device, mat, size, cudaMemcpyHostToDevice);

    size_t size1 = cols*sizeof(float);
    float *mean = (float *)malloc(size1);
    float *mean_device = nullptr;
    cudaMalloc((void **)&mean_device,size1);
    cudaMemcpy(mean_device, mean, size1, cudaMemcpyHostToDevice);
    

    float *std = (float *)malloc(size1);
    float *std_device = nullptr;
    cudaMalloc((void **)&std_device, size1);
    cudaMemcpy(std_device, std, size1, cudaMemcpyHostToDevice);
  

  //  auto t0 = high_resolution_clock::now();
    mean_by_rows_cpu(mat_device,mean_device, rows,cols);
    normalization_by_rows_cpu(mat_device,mean_device,std_device, rows,cols);
    cudaDeviceSynchronize();
  //  auto t1 = high_resolution_clock::now();

    cudaMemcpy(mat, mat_device, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(mean, mean_device, size1, cudaMemcpyDeviceToHost);
    cudaMemcpy(std, std_device, size1, cudaMemcpyDeviceToHost);

//
    for(u32 i=0; i<20;i++){
        printf("val %d %f %f %f\n",i,mat[i], mean[i], std[i]);
    }

    cudaFree(mean_device);
    cudaFree(mat_device);
    cudaFree(std_device);
   // printf("take time %d\n",duration_cast<milliseconds>(t1-t0).count());
    free(mat);
    free(mean);
    free(std);
    return 0;
}

