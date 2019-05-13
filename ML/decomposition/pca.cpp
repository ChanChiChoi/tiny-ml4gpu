#include "ML/preprocessing/data.h"
#include "common/buffer_info_ex.h"
#include "ML/math/math.h"
#include "common/type.h"
#include "ML/decomposition/pca.h"
#include "common/malloc_free.h"


PCA&
PCA::fit(Array &matrix){

    float * ptr_device = (float *)(matrix.ptr_buf->ptr_device);

    //1 - substract the mean by sample
    size_t rows = matrix.ptr_buf->shape[0];
    size_t cols = matrix.ptr_buf->shape[1];
    size_t size_mean = sizeof(float)*cols;

    float *mean = (float *)malloc(size_mean);
    float *mean_device = HOST_TO_DEVICE_MALLOC(mean, size_mean);

    delete mean_vec;
    mean_vec = new Array{
                nullptr,  mean, mean_device,
                2, {1, ssize_t(cols)}, std::string(1,'f'), 
                ssize_t(sizeof(float)), ssize_t(1*cols),
                {ssize_t(sizeof(float)*cols),ssize_t(sizeof(float))}
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
    ci32 Row_V = Col_VT, Col_V = Row_VT;
    
    size_t size_U = sizeof(float)*Row_U*Col_U;
    size_t size_S = sizeof(float)*Length;
    size_t size_VT = sizeof(float)*Row_VT*Col_VT;

    float *U = (float *)malloc(size_U);
    float *S = (float *)malloc(size_S);
    float *V = (float *)malloc(size_VT); //for V not VT

    float *U_device = HOST_TO_DEVICE_MALLOC(U, size_U);
    float *S_device = HOST_TO_DEVICE_MALLOC(S, size_S);
    float *VT_device = HOST_TO_DEVICE_MALLOC(V, size_VT);
    
    svd(mat_cov_device, Row_A, Col_A, lda,
        U_device, Row_U, Col_U,
        S_device, Length,
        VT_device, Row_VT, Col_VT);

    // copy U ,S to host then free
    DEVICE_TO_HOST_FREE(U,U_device, size_U);
    DEVICE_TO_HOST_FREE(S,S_device, size_S);
    free(U);
    free(S);

    //delete mat_cov matrix,
    DEVICE_TO_HOST_FREE(mat_cov, mat_cov_device, size_cov);
    free(mat_cov);

    // get V ,not VT
    float *V_device;
    V_device = DEVICE_MALLOC(V_device, size_VT);
    matrix_transpose_cpu(VT_device,Row_VT, Col_VT,
                         V_device, Row_V, Col_V);
    DEVICE_FREE(VT_device);
    DEVICE_TO_HOST(V, V_device, size_VT);

    delete trans_mat;
    trans_mat = new Array{
                nullptr, V, V_device,
                2, {ssize_t(Row_V), ssize_t(Col_V)}, std::string(1,'f'),
                ssize_t(sizeof(float)), ssize_t(Row_V*Col_V),
                {ssize_t(sizeof(float)*Col_V), ssize_t(sizeof(float))}
                }; // need parameter
    

}

Array *
PCA::transform(Array &matrix){
    // first min(m,n) columns of U and V are left and right singular vectors of A
  
    assert(matrix.ptr_buf->ptr_device != nullptr);
    float * ptr_device = (float *)matrix.ptr_buf->ptr_device;
    size_t Row_mat = matrix.ptr_buf->shape[0];
    size_t Col_mat = matrix.ptr_buf->shape[1]; 
    assert(Col_mat == trans_mat->ptr_buf->shape[1]); 

    // step 1: subtract the mean from matrix
    mean_by_rows_cpu((float *)ptr_device, (float *)(mean_vec->ptr_buf->ptr_device), Row_mat, Col_mat);
    // step 2: get n_components col of V

    float * V_device = (float *)(trans_mat->ptr_buf->ptr_device);
    size_t Row_V = trans_mat->ptr_buf->shape[0];
    size_t Col_V = trans_mat->ptr_buf->shape[1];

    // malloc for result
    size_t Row_ans = Row_mat, Col_ans = n_components;
    size_t size_ans = sizeof(float)*Row_ans*Col_ans;
    float *ans = (float *)malloc(size_ans); 
    float *ans_device = HOST_TO_DEVICE_MALLOC(ans,size_ans);

    // get submatrix from V
    float *tmp_device;
    size_t Row_tmp = Row_V, Col_tmp = n_components;
    size_t size_tmp = sizeof(float)*Row_tmp*Col_tmp;
    tmp_device = DEVICE_MALLOC(tmp_device, size_tmp);
    
    matrix_subblock_cpu(V_device, Row_V, Col_V,
                       tmp_device, Row_tmp, Col_tmp,
                       0, 0, Row_V, Col_tmp);

    // matrix multiply the V
    matrix_mul_cpu(ptr_device,Row_mat, Col_mat,
                   tmp_device, Row_tmp, Col_tmp,
                   ans_device, Row_ans, Col_ans);

    DEVICE_FREE(tmp_device);

    DEVICE_TO_HOST(ans, ans_device, size_ans); 
    // get first n_components columns: [d,n_components]
    // res = matrix x ans: n*d times d*n_components
    
    return new Array{
            nullptr, ans, ans_device,
            2, {ssize_t(Row_ans), ssize_t(Col_ans)}, std::string(1,'f'),
            ssize_t(sizeof(float)), ssize_t(Row_ans*Col_ans),
            {ssize_t(sizeof(float)*Col_ans), ssize_t(sizeof(float))}
            };
    
}

/* from ml4gpu import Array,PCA
 * dataset = Array(np.dataset)
 * dataset.cuda()
 * pca = PCA(3);
 * pca.fit(dataset)
 * gpu.data = pca.transform(dataset1)
 * cpu.data = gpu.data.cpu()
 * */
