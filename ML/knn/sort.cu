#include <stdio.h>
#include <assert.h>
#include <limits>
#include <cuda_runtime.h>
#include "common/malloc_free.h"
#include "ML/knn/sort.h"

typedef unsigned int u32;
#define MAX_NUM_LISTS 128


// radix sort only support unsigned int.
// when handling float, we can `unsigned int after = (unsigned int)(float before*1000)`

template<typename T> __device__ void
radix_sort2(T  * const sort_tmp,
            u32  * sort_ind,
            const u32  num_lists,
            const u32  num_elements,
            const u32  tid,
            T  * const sort_tmp_1,
            u32  *sort_ind_1,
            u32 precision = 1,
            u32 bit_size = 32){

    // num_lists must be even
    assert(num_lists % 2 == 0);

    // init the ind vector
    u32 i_tid = 0+tid;
    for(u32  i = 0;i+tid < num_elements; i+= num_lists){
        i_tid = i+tid;
        sort_ind[i_tid] = i_tid;
    }
    

    for (u32  bit = 0; bit < bit_size; bit++){

        const u32  bit_mask = (1 << bit);
        u32  base_cnt_0 = 0;
        u32  base_cnt_1 = 0;


        i_tid = 0+tid;
        for(u32  i = 0;  i+tid < num_elements; i+= num_lists){

          i_tid = i+tid;
          // get the val, then if float, we should preserve precision,e.g, 1 10 100 1000
          const T elem_tmp = sort_tmp[i_tid];
          // radix sort only support unsigned int
          const u32 elem = (u32)(elem_tmp*precision);
//          const u32  elem = sort_tmp[i_tid];

//          if(tid == 0 && bit == 1)
//              printf(" [%f %d] ",elem_tmp, elem );
          const u32  ind = sort_ind[i_tid];

          if ((elem & bit_mask) > 0){
              sort_tmp_1[base_cnt_1+tid] = elem_tmp;
              // handle the index
              sort_ind_1[base_cnt_1+tid] = ind;
              base_cnt_1 += num_lists;
          }else{
              sort_tmp[base_cnt_0+tid] = elem_tmp;
              // handle the index
              sort_ind[base_cnt_0+tid] = ind;
              base_cnt_0 += num_lists;
          }
        }

        // copy data back to source from the one's list 
        /*cannot use sort_ind replace sort_tmp_1, because after some iter,
        the value of the ind has not been the origin value.
        */
        i_tid = 0+tid;
        for(u32  i = 0; i<base_cnt_1; i += num_lists){
            i_tid = i+tid;
            sort_tmp[base_cnt_0+i_tid] = sort_tmp_1[i_tid];
            sort_ind[base_cnt_0+i_tid] = sort_ind_1[i_tid];
        }
    }

}


template<typename T> __device__ void
merge_array(const T * const src_array,
            const u32 * const src_ind_array,
            T * const dest_array,
            u32 * const dest_ind_array,
            const u32 num_lists,
            const u32 num_elements,
            const u32 tid,
            const T max_val ){

    // num_lists must be even
    assert(num_lists % 2 == 0);
    //const u32 num_elements_per_list = num_elements / num_lists;
  
    __shared__ u32 list_indexes[MAX_NUM_LISTS];
    __shared__ T reduction_val[MAX_NUM_LISTS];
    __shared__ u32 reduction_idx[MAX_NUM_LISTS];

    // 1 - clear the working set
    list_indexes[tid] = 0; // current tid had handled elems
    reduction_val[tid] = 0;
    reduction_idx[tid] = 0;
    __syncthreads();

    for(u32 i = 0; i < num_elements; i++){

       u32 tid_max = num_lists >> 1;   
       T data;

       // 2 - for current thread, get data
       // whether current tid has handle the num of elems
       //if (list_indexes[tid] < num_elements_per_list){
       if (tid+list_indexes[tid]*num_lists < num_elements){
           // cur data index in src array
           const u32 src_idx = tid + (list_indexes[tid] * num_lists);
           data = src_array[src_idx];
       }else{
           data = max_val; // data = 0xFFFFFFFF;
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
               const T val2 = reduction_val[val2_idx];

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


template<typename T> __global__ void
sort_by_rows(T  *mat, u32  *ind_mat, size_t rows, size_t cols, 
             T  * tmp_1, u32  *ind_1, u32 num_lists, u32 precision, T max_val){

    //num_lists should be 256;
    u32 bx = blockIdx.x;
    u32 tx = threadIdx.x;

    radix_sort2<T>(mat+bx*cols, ind_mat+bx*cols,
              num_lists,cols,tx,
              tmp_1+bx*cols, 
              ind_1+bx*cols ,
              precision);
        
    __syncthreads();


    merge_array<T>(mat+bx*cols,ind_mat+bx*cols,
                tmp_1+bx*cols, ind_1+bx*cols,
                num_lists,cols,tx, max_val);
}


template<typename T> void
sort_by_rows_cpu(T  *mat, u32  *ind_mat, size_t rows, size_t cols, u32 precision, T max_val){
    
    size_t size = sizeof(T)*rows*cols;
    size_t size1 = sizeof(u32)*rows*cols;

    // 2function
    T *mat_d = host_to_device_malloc(mat, size);
    u32 *ind_mat_d = host_to_device_malloc(ind_mat, size1);

    // result of two buffer
    T *tmp_1 = device_malloc<T>(size);
    u32 *ind_1 = device_malloc<u32>(size1);
     
    u32 num_lists = MAX_NUM_LISTS;
    dim3 grid(rows);
    dim3 block(num_lists);
    sort_by_rows<T><<<grid,block>>>(mat_d, ind_mat_d, rows, cols,tmp_1,ind_1,num_lists, precision, max_val);


    //2 function
    device_free<T>(mat_d);
    device_free<u32>(ind_mat_d);

    device_to_host_free(mat, tmp_1, size);
    device_to_host_free(ind_mat, ind_1, size1);

//    printf("======================\n");
//    for(int i = 0; i<cols;i++){
//       printf(" [%f %d] ",mat[i],ind_mat[i]);
//    }
} 

template<typename T> void
sort_by_rows_gpu(T  *mat_d, u32  *ind_mat_d, size_t rows, size_t cols, u32 precision, T max_val){

    size_t size = sizeof(T)*rows*cols;
    size_t size1 = sizeof(u32)*rows*cols;


    // result of two buffer
    T *tmp_1 = device_malloc<T>(size);
    u32 *ind_1 = device_malloc<u32>(size1);

    u32 num_lists = MAX_NUM_LISTS;
    dim3 grid(rows);
    dim3 block(num_lists);
    sort_by_rows<T><<<grid,block>>>(mat_d, ind_mat_d, rows, cols,tmp_1,ind_1,num_lists, precision, max_val);


    device_to_device(mat_d, tmp_1, size);
    device_to_device(ind_mat_d, ind_1, size1);

    device_free<T>(tmp_1);
    device_free<u32>(ind_1);

}

void
sort_by_rows(float *mat, u32 *ind_mat, size_t rows, size_t cols, u32 precision){

    float max_val = std::numeric_limits<float>::max();
    sort_by_rows_gpu<float>(mat, ind_mat, rows, cols, precision, max_val);
}


void
sort_by_rows(u32 *mat, u32 *ind_mat, size_t rows, size_t cols, u32 precision){
    u32 max_val = std::numeric_limits<u32>::max();
    sort_by_rows_gpu<u32>(mat, ind_mat, rows, cols, precision, max_val);
    
}


//int
//main(){
//
//    size_t cols = 200;
//    size_t  rows = 1;
//    size_t size = sizeof(u32 )*cols*rows;
//    float  *mat = (float  *)malloc(size);
//
//    u32  *ind_mat = (u32  *)malloc(size);
//
//    for(int i=0; i<cols; i++){
//        mat[i] = cols-i;
//       // mat[i+cols] = cols-i;
//    }
//    float *mat_d = host_to_device_malloc(mat,size);
//    u32 *ind_mat_d = host_to_device_malloc(ind_mat,size);
//
//    u32 precision = 1;
//    float max_val = std::numeric_limits<float>::max();
//    
//    sort_by_rows_gpu<float>(mat_d, ind_mat_d, rows, cols, precision,max_val);
//
//
//    device_to_host_free(mat,mat_d,size);
//    device_to_host_free(ind_mat,ind_mat_d,size);
//    for(int i=0; i<cols; i++){
//        printf("mat %f\n",mat[i]);
//    }
//    free(mat);
//    free(ind_mat);
//
//}
