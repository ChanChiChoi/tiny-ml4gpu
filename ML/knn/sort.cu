#include<stdio.h>
#include<cuda_runtime.h>
#include "common/malloc_free.h"

typedef unsigned int u32;
__device__ void
radix_sort2(unsigned int * const sort_tmp,
            unsigned int * sort_ind,
            const unsigned int num_lists,
            const unsigned int num_elements,
            const unsigned int tid,
            unsigned int * const sort_tmp_1,
            unsigned int *sort_ind_1){

    // num_lists must be even
    assert(num_lists % 2 == 0);

    // init the ind vector
    for(unsigned int i = 0; i+tid< num_elements; i+= num_lists){
        sort_ind[i+tid] = i+tid;
    }
    

    for (unsigned int bit = 0; bit < 32; bit++){

        const unsigned int bit_mask = (1 << bit);
        unsigned int base_cnt_0 = 0;
        unsigned int base_cnt_1 = 0;

        for(unsigned int i = 0;  i+tid < num_elements; i+= num_lists){
          //const unsigned int elem = (unsigned int)(sort_tmp[i+tid]*100);
          const unsigned int elem = sort_tmp[i+tid];
          if(bit == 0 && tid==9)
          printf(" [i+tid %d, i %d, tid %d] ",i+tid, i,tid);

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

}


__device__ void
merge_array(const u32 * const src_array,
            const u32 * const src_ind_array,
            u32 * const dest_array,
            u32 * const dest_ind_array,
            const u32 num_lists,
            const u32 num_elements,
            const u32 tid ){

    // num_lists must be even
    assert(num_lists % 2 == 0);
    //const u32 num_elements_per_list = num_elements / num_lists;
  
    __shared__ u32 list_indexes[MAX_NUM_LISTS];
    __shared__ u32 reduction_val[MAX_NUM_LISTS];
    __shared__ u32 reduction_idx[MAX_NUM_LISTS];

    // 1 - clear the working set
    list_indexes[tid] = 0; // current tid had handled elems
    reduction_val[tid] = 0;
    reduction_idx[tid] = 0;
    __syncthreads();

    for(u32 i = 0; i < num_elements; i++){

       u32 tid_max = num_lists >> 1;   
       u32 data;

       // 2 - for current thread, get data
       // whether current tid has handle the num of elems
       //if (list_indexes[tid] < num_elements_per_list){
       if (tid+list_indexes[tid]*num_lists < num_elements){
           // cur data index in src array
           const u32 src_idx = tid + (list_indexes[tid] * num_lists);
           data = src_array[src_idx];
       }else{
           data = 0xFFFFFFFF;
       }

       //store the current data value and index
       reduction_val[tid] = data;
       reduction_idx[tid] = tid;

       // wait for all threads to copy
       __syncthreads();

       
       // 3 - reduce from num_lists to one thread zero
       while(tid_max != 0){
           // gradually reduce tid_max from num_lists to zero

           if(tid < tid_max){
               // calculate the index of  the other half
               // the id of other thread
               const u32 val2_idx = tid + tid_max;
               // read in the other half
               const u32 val2 = reduction_val[val2_idx];

               //if this half is bigger
               if (reduction_val[tid] > val2){
                   // the store the smaller value
                   reduction_val[tid] = val2;
                   reduction_idx[tid] = reduction_idx[val2_idx];
               }

           }

           // divide tid_max by two
           tid_max >>= 1;

           __syncthreads();
       }

       // 4 - only 0 ind can store dest value
       if (tid == 0){

           // store the winning value
           dest_array[i] = reduction_val[0];

           const u32 ind_idx = list_indexes[reduction_idx[0]]*num_lists + reduction_idx[0];
           dest_ind_array[i] = src_ind_array[ind_idx];

           // increment the list pointer for this thread
           list_indexes[reduction_idx[0]] ++ ;
       }
       
       // wait for tid zero
       __syncthreads();

    }

}


__global__ void
sort_by_rows(unsigned int *mat, unsigned int *ind_mat, size_t rows, size_t cols, 
             unsigned int * tmp_1, unsigned int *ind_1, unsigned int num_lists){

    //num_lists should be 256;
    unsigned int bx = blockIdx.x;
    unsigned int tx = threadIdx.x;

    radix_sort2(mat+bx*cols, ind_mat+bx*cols,
              num_lists,cols,tx,
              tmp_1+bx*cols, ind_1+bx*cols );
        
    __syncthreads();


    merge_array(mat+bx*cols,ind_mat+bx*cols,
                tmp_1+bx*cols, ind_1+bx*cols,
                num_lists,cols,tx);
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
     
    unsigned int num_lists = 128;
    dim3 grid(rows);
    dim3 block(num_lists);
    sort_by_rows<<<grid,block>>>(mat_d, ind_mat_d, rows, cols,tmp_1,ind_1,num_lists);

    //result in tmp_1 and ind_1
    device_free<unsigned int>(tmp_1);
    device_free<unsigned int>(ind_1);    

    //2 function
    device_to_host(mat_d, mat, size);
    for(int i = 0; i<cols;i++){
       printf("%d ",mat[i]);
    }
//    for(int i = 0; i<cols;i++){
//       printf("%d ",mat[i+cols]);
//    }
    device_to_host(ind_mat_d, ind_mat, size1);
} 


int
main(){

    size_t cols = 200;
    size_t  rows = 1;
    size_t size = sizeof(unsigned int)*cols*rows;
    unsigned int *mat = (unsigned int *)malloc(size);

    unsigned int *ind_mat = (unsigned int *)malloc(size);

    for(int i=0; i<cols; i++){
        mat[i] = cols-i;
       // mat[i+cols] = cols-i;
    }
    sort_by_rows_cpu(mat, ind_mat, rows, cols);

    free(mat);
    free(ind_mat);

}
