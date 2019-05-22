#include "vector.cu"

int
main(){
     
    size_t rows = 10000;
    size_t cols = 10000;
    size_t size = sizeof(float)*rows*cols;
    float *mat = (float *)malloc(size);

    float *md_device;
    cudaMalloc((void **)&md_device,size);
    cudaMemcpy(md_device, mat, size, cudaMemcpyHostToDevice);


    size_t cols_vec = cols;
    size_t size1 = sizeof(float)*cols_vec;
    float *vec = (float *)malloc(size1);

    float *vec_device;
    cudaMalloc((void **)&vec_device,size1);
    cudaMemcpy(vec_device, vec, size1, cudaMemcpyHostToDevice);

    vector_repeat_by_rows_cpu(md_device, rows, cols,
                vec_device, cols_vec);
    cudaFree(md_device);
    cudaFree(vec_device);
    free(mat);
    free(vec);
    return 0;
}
