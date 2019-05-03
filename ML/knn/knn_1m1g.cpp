#include "common/buffer_info_ex.h"
#include <pybind11/pybind11.h>
#include "ML/knn/matrix.cuh"
#include "common/malloc_free.h"

namespace py = pybind11;

class KNN{


    Array *train = NULL;
    Array *labels = NULL;
    
    Array *ans = NULL;

public:

    KNN() {}

    KNN(Array &array){

    }

    // need overload 2 case: i)numpy -> fit; ii)numpy -> pre -> fit
//    KNN & fit(py::array_t<float> &array);

    KNN & fit(Array &train, Arrat &labels);

    py::array_t<float> predict(Array &test);
};



KNN & KNN::fit(Array &train, Array &labels){
    train = &train;
    labels = &labels;
}

py::array_t<float> KNN::predict(Array &test, size_t k){
    // because the function will return result into python
    size_t ans_rows = test->shape[0];
    size_t ans_cols = train->shape[0];
    size_t bytes_size = sizeof(float)*ans_rows*ans_cols;
    
    float *ans = (float *) malloc(bytes_size);
    float *ans_device = host_to_device(ans, bytes_size);

    // train matrix multi test matrix
    matrix_mul_cpu((float*)train->ptr_device, train->shape[0], train->shape[1],
                   (float*)test->ptr_device, test->shape[0], test->shape[1],
                   ans_device, ans_rows,ans_cols); 

  
    // sort by distance ,then extract the index
    unsigned int *ind_mat = (unsigned int *)malloc(bytes_size);
    unsigned int *ind_mat_device =  host_to_device(ind_mat, bytes_size);

    sort_by_rows_k(ans_device, ind_mat, ans_rows, ans_cols, k);


    free(ans);
    free(ind_mat);
    return  ; 
}

/*
 * from ml4gpu import KNN,Array
 * knn = KNN()
 * train,labels = Array(np.train),Array(np.labels)
 * train.cuda()
 * labels.cuda()
 * knn.fit(train,labels)
 * test = Array(np.test)
 * test.cuda()
 * ans = knn.pred(test)
 * train.cpu() // release the memory
 * test.cpu()
 * -------------------------------
 * from ml4gpu import KNN,dataPre
 * norData = dataPre()
 * gpu.train = norData.normal(np.train)
 *
 * knn = KNN()
 * knn.fit(gpu.train)
 * gpu.test = norData.transform(np.test)
 * ans = knn.pred(gpu.test)
 * */
