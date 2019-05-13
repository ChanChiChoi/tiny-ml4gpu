#include <cuda_runtime.h>
#include "ML/preprocessing/data.h"
#include "common/buffer_info_ex.h"
#include "ML/math/math.h"
#include "common/type.h"

class PCA{

    Array *trans_mat = NULL;
    Array *mean_vec = NULL;
    size_t n_components = 0;

public:
    PCA (){}

    // only init the n_components;
    PCA ( size_t n_components):n_components{n_components}{
        trans_mat = new Array();
        mean_vec = new Array();
    }

    // will stat the matrix, then put the transfer matrix into trans_mat
    PCA & fit(Array &matrix);

    Array & transform(Array &matrix);

    ~PCA(){
        if (trans_mat){
            delete trans_mat;
            trans_mat = NULL;
        }
        if (mean_vec){
            delete mean_vec;
            mean_vec = NULL;
        }
    }

};



PCA&
PCA::fit(Array &matrix){

    float * ptr_device = (float *)matrix.ptr_device;

    //1 - substract the mean by sample
    size_t rows = matrix.shape[0];
    size_t cols = matrix.shape[1];
    size_t size_mean = sizeof(float)*cols;

    float *mean = (float *)malloc(size_mean);
    float *mean_device = HOST_TO_DEVICE_MALLOC(mean, size_mean);

    delete mean_vec;
    mean_vec = new Array{
                nullptr,  mean, mean_device,
                2, {1, Col_VT}, std::string(1,'f'), 
                sizeof(float), 1*cols,
                {sizeof(float)*cols,sizeof(float)}
                }; // need parameter
    
    //TODO: we need keep mean_device
    mean_by_rows_cpu(ptr_device, mean_device, rows, cols);
    DEVICE_TO_HOST(mean, mean_device, size_mean);

    //2 - calc the cov matrix
    size_t rows_cov = cols;
    size_t cols_cov = cols;
    size_t size_cov = sizeof(float)*rows_cov*cols_cov;
    
    float *mat_cov = (float *)malloc(size_cov);
    float *mat_cov_device = HOST_TO_DEVICE_MALLOC(mat_cov, size_cov);
    
    cov_cpu(ptr_device, rows, cols,
            mat_cov_device, rows_cov, cols_cov );

    //3 - use svd to calc the eigval and eigvec
    // A = U*S*VT
    ci32 Row_A = rows_cov, Col_A = cols_cov;
    ci32 lda = Row_A;
    ci32 Row_U = lda, Col_U = lda; //    
    ci32 Length = Col_A; //
    ci32 Row_VT = Col_A, Col_VT = Col_A;
    
    size_t size_U = sizeof(float)*Row_U*Col_U;
    size_t size_S = sizeof(float)*Length;
    size_t size_VT = sizeof(float)*Row_VT*Col_VT;

    float *U = (float *)malloc(size_U);
    float *S = (float *)malloc(size_S);
    float *VT = (float *)malloc(size_VT);

    float *U_device = HOST_TO_DEVICE_MALLOC(U, size_U);
    float *S_device = HOST_TO_DEVICE_MALLOC(S, size_S);
    float *VT_device = HOST_TO_DEVICE_MALLOC(VT, size_VT);
    
    svd(mat_cov_device, Row_A, Col_A, lda,
        U_device, Row_U, Col_U,
        S_device, Length,
        VT_device, Row_VT, Col_VT);

    // copy U ,S to host then free
    DEVICE_TO_HOST_FREE(U,U_device, size_U);
    DEVICE_TO_HOST_FREE(S,S_device, size_S);
    free(U);
    free(S);

    //6 - hold the VT matrix, then that is all

    DEVICE_TO_HOST_FREE(mat_cov, mat_cov_device, size_cov);
    free(mat_cov);

    delete trans_mat;
    trans_mat = new Array{
                nullptr, VT, VT_device,
                2, {Row_VT, Col_VT}, std::string(1,'f'),
                sizeof(float), Row_VT*Col_VT,
                {sizeof(float)*Col_VT, sizeof(float)}
                }; // need parameter
    

}

Array &
PCA::transform(Array &matrix){
  
    assert(matrix->ptr_buf->ptr_device != nullptr);
    
}

/* from ml4gpu import Array,PCA
 * dataset = Array(np.dataset)
 * dataset.cuda()
 * pca = PCA(3);
 * pca.fit(dataset)
 * gpu.data = pca.transform(dataset1)
 * cpu.data = gpu.data.cpu()
 * */
