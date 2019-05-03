#include<stdio.h>
#include<cuda_runtime.h>
#include "common/malloc_free.h"

__device__ void
radix_sort2(unsigned int * const sort_tmp,
            unsigned int * sort_ind,
            const unsigned int num_lists,
            const unsigned int num_elements,
            const unsigned int tid,
            unsigned int * const sort_tmp_1,
            unsigned int *sort_ind_1){

    // init the ind vector
    for(unsigned int i = 0; i< num_elements; i+= num_lists){
        sort_ind[i+tid] = i+tid;
    }
    __syncthreads();
    

    for (unsigned int bit = 0; bit < 32; bit++){

        const unsigned int bit_mask = (1 << bit);
        unsigned int base_cnt_0 = 0;
        unsigned int base_cnt_1 = 0;

        for(unsigned int i = 0; i< num_elements; i+= num_lists){
          //const unsigned int elem = (unsigned int)(sort_tmp[i+tid]*100);
          const unsigned int elem = sort_tmp[i+tid];

          const unsigned int ind = sort_ind[i+tid];
          if ((elem & bit_mask) > 0){
              sort_tmp_1[base_cnt_1+tid] = elem;
              // handle the index
              sort_ind_1[base_cnt_1+tid] = ind;
              base_cnt_1 += num_lists;
          }else{
              sort_tmp[base_cnt_0+tid] = elem;
              // handle the index
              sort_ind[base_cnt_0+tid] = ind;
              base_cnt_0 += num_lists;
          }
        }
        // copy data back to source from the one's list 
        for(unsigned int i = 0; i<base_cnt_1; i += num_lists){
            sort_tmp[base_cnt_0+i+tid] = sort_tmp_1[i+tid];
            sort_ind[base_cnt_0+i+tid] = sort_ind_1[i+tid];
        }
    }
    __syncthreads();

}


__global__ void
sort_by_rows(unsigned int *mat, unsigned int *ind_mat, size_t rows, size_t cols, 
             unsigned int * tmp_1, unsigned int *ind_1, unsigned int num_lists){

    //num_lists should be 256;
    unsigned int bx = blockIdx.x;
    unsigned int tx = threadIdx.x;
    size_t size = sizeof(unsigned int)*cols;
    radix_sort2(mat+bx*cols, ind_mat+bx*cols,
              num_lists,cols,tx,
              tmp_1+bx*cols, ind_1+bx*cols );
        
    __syncthreads();

}



void
sort_by_rows_cpu(unsigned int *mat,unsigned int *ind_mat, size_t rows, size_t cols){
    
    size_t size = sizeof(unsigned int)*rows*cols;
    size_t size1 = sizeof(unsigned int)*rows*cols;

    // 2function
    unsigned int *mat_d = host_to_device(mat, size);
    unsigned int *ind_mat_d = host_to_device(ind_mat, size1);



    unsigned int *tmp_1 = device_malloc<unsigned int>(size);
    unsigned int *ind_1 = device_malloc<unsigned int>(size1);
     
    unsigned int num_lists = 256;
    dim3 grid(rows);
    dim3 block(num_lists);
    sort_by_rows<<<grid,block>>>(mat_d, ind_mat_d, rows, cols,tmp_1,ind_1,num_lists);

    device_free(tmp_1);
    device_free(ind_1);    

    //2 function
    device_to_host(mat_d, mat, size);
//    for(int i = 0; i<cols;i++)
//       printf("%ld ",mat[i]);
    device_to_host(ind_mat_d, ind_mat, size1);
} 
