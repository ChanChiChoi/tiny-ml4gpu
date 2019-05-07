#include <cuda_runtime.h>
#include "ML/preprocessing/data.h"
#include "common/buffer_info_ex.h"

class PCA{

    Array &trans_mat;
    size_t n_components;

public:
    PCA (){}

    // only init the n_components;
    PCA ( size_t n_components):n_components{n_components}{}

    // will stat the matrix, then put the transfer matrix into trans_mat
    PCA & fit(Array &matrix);

    PCA & transform(Array &matrix);

};

PCA&
PCA::fit(Array &matrix){

    auto ptr_device = matrix.ptr_device;
    //1 - substract the mean by sample
    size_t rows = matrix.shape[0];
    size_t cols = matrix.shape[1];
    size_t size = sizeof(float)*cols;

    float *mean = (float *)malloc(size);
    float *mean_device = host_to_device_malloc(mean, size);

    mean_by_rows_cpu(ptr_device, mean_device, rows, cols);
    //2 - calc the cov matrix
    //3 - use svd to calc the eigval and eigvec
    //4 - sort the eigval
    //5 - preserve the first nth eigval
    //6 - use eigvec to transfer the origin matrix into new space.


}

/* from ml4gpu import Array,PCA
 * dataset = Array(np.dataset)
 * dataset.cuda()
 * pca = PCA(3);
 * pca.fit(dataset)
 * gpu.data = pca.transform(dataset)
 * cpu.data = gpu.data.cpu()
 * */
