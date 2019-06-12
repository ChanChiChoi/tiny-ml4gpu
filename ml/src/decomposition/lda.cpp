/* borrowde from https://github.com/UMD-ISL/Matlab-Toolbox-for-Dimensionality-Reduction/blob/master/techniques/lda.m
 *
 * https://www.cnblogs.com/shouhuxianjian/p/9773200.html
 * */
#include <string>
#include <vector>
#include "ml/include/decomposition/lda.h"
#include "ml/include/math/math.h"
#include "ml/include/preprocessing/preprocess.h"
#include "common/include/type.h"
#include "common/include/buffer_info_ex.h"
#include "common/include/malloc_free.h"

using std::vector;

void
stat_ind(vector<vector<int>> &ans, int *vec, size_t num){

    for(int i=0; i<num; i++){
        ans[vec[i]].push_back(i);
    }
}

Array * 
LDA::fit(Array &matrix, Array &labels, size_t n_classes){

    /*
 *  labels should start with 1, not 0;
 *  */
    float *train_device = (float *)matrix.ptr_buf->ptr_device;
    assert(matrix.ptr_buf->ndim == 2);
    size_t rows = matrix.ptr_buf->shape[0];
    size_t cols = matrix.ptr_buf->shape[1];

    // computer the mean by rows
    size_t size_mean = sizeof(float)*cols;
    float *mean = (float *)malloc(size_mean);
    float *mean_device = HOST_TO_DEVICE_MALLOC(mean_device, size_mean);

    // substract mean from matrix
    subtract_mean_by_rows_cpu(train_device, mean_device, rows, cols);
    DEVICE_TO_HOST(mean, mean_device, size_mean);
    
    delete this->mean_vec;
    this->mean_vec = new Array{
          nullptr, mean, mean_device,
          2, {ssize_t(1), ssize_t(cols)}, std::string(1,'f'),
          ssize_t(sizeof(float)), ssize_t(1*cols),
          {ssize_t(sizeof(float)*cols), ssize_t(sizeof(float))}
        };

    // intialize Sw
    // Sw = zeros(size(X, 2), size(X, 2));
    float *Sw = nullptr;
    size_t size_sw = sizeof(float)*cols*cols;
    Sw = DEVICE_MALLOC(Sw,size_sw);
    DEVICE_MEMSET(Sw, 0, size_sw);
 
    // compute total convariance matrix
    // St = cov(X);
    float *St = nullptr;
    size_t size_st = sizeof(float)*cols*cols;
    St = DEVICE_MALLOC(St, size_st);
    cov_cpu(train_device, rows, cols,
            St, cols, cols);

    // sum over classes, 
    // get all instances with ith class
    // Sw += p*C;
    int *labels_host = (int *)labels.ptr_buf->ptr;
    size_t num = labels.ptr_buf->size;
    vector<vector<int>> ind_vec(n_classes+1, vector<int>());

    /* if labels start with 0, then [n_classes+1] vector has 0 size; */
    /* else start with 1, then [0] vector has 0 size; */
    stat_ind(ind_vec, labels_host, num);

    // compute between class scatter
    // Sb = St - Sw;
    int *ind_device = nullptr;
    int length = 0;
    float *matrix_sub = nullptr;
    float *mat_C = nullptr;
    float p = 0.0;
    for (int i=0; i<n_classes+1; i++){
       length = ind_vec[i].size();
       if (length == 0){
           continue;
       }
       
       ind_device = DEVICE_MALLOC(ind_device, sizeof(int)*length);
       HOST_TO_DEVICE(ind_device, &ind_vec[i][0], sizeof(int)*length);
       // call matrix_sub_by_row
       // cur_X = X(labels == i,:);
       matrix_sub = DEVICE_MALLOC(matrix_sub, sizeof(float)*cols*length);
       matrix_sub_by_rows_cpu(train_device, rows, cols,
                              matrix_sub, length, cols,
                              ind_device, length);

       // C = cov(cur_X);
       mat_C = DEVICE_MALLOC(mat_C, sizeof(float)*cols*cols);
       cov_cpu(matrix_sub, length, cols,
               mat_C, cols, cols );

       // p = size(cur_X, 1) / (length(labels) - 1);
       p = length / (n_classes-1); 
       // Sw = Sw + (p * C);
       matrix_mul_scalar_cpu(mat_C, rows, cols, p);
       matrix_mul_cpu(Sw, cols, cols,
               mat_C, cols, cols,
               Sw, cols, cols,
               SCALAR_TWO_ADD);
    
       DEVICE_FREE(ind_device);
       DEVICE_FREE(matrix_sub);
       DEVICE_FREE(mat_C);
    }   

    //  Sb       = St - Sw;
    size_t size_sb = sizeof(float)*cols*cols;
    float *Sb = nullptr;
    Sb = DEVICE_MALLOC(Sb, size_sb);
    matrix_mul_cpu(St, cols, cols,
                   Sw, cols, cols,
                   Sb, cols, cols,
                   SCALAR_TWO_DIVIDE);
    /* Sb(isnan(Sb)) = 0; Sw(isnan(Sw)) = 0; */
    /* Sb(isinf(Sb)) = 0; Sw(isinf(Sw)) = 0; */

    // make sure not to embed in too high dimension
    // no_dims = nc - 1;
    if (n_classes <= this->n_components){
        this->n_components = n_classes - 1;
    }
    // perform engendecomposition of inv(Sw)*Sb;
    // eig(Sb, Sw);

    // sort engenvalue and eigen vectors in descending order

    // compute mapped data
    // mappedX = X*M;

    // store mapping for the out-of-sample extension
    // mapping.M = M;
    // mapping.val = lambda;
}


Array * 
KPCA::transform(Array &train, Array &test){
/*
 * % Compute and center kernel matrix
 * K = gram(mapping.X, point, mapping.kernel, mapping.param1, mapping.param2);
 * J = repmat(mapping.column_sums', [1 size(K, 2)]);
 * K = K - repmat(sum(K, 1), [size(K, 1) 1]) - J + repmat(mapping.total_sum, [size(K, 1) size(K, 2)]);
 *                                                 
 * % Compute transformed points
 * t_point = mapping.invsqrtL * mapping.V' * K;
 * t_point = t_point';
 *
 * */
    //1 - gram

    float *X_device = (float *)train.ptr_buf->ptr_device;
    float *point_device = (float *)test.ptr_buf->ptr_device;
    // get point_T_device

    size_t Row_X = train.ptr_buf->shape[0];
    size_t Col_X = train.ptr_buf->shape[1];

    size_t Row_point = test.ptr_buf->shape[0];
    size_t Col_point = test.ptr_buf->shape[1];


    float *point_T_device = nullptr;
    size_t Row_point_T = Col_point, Col_point_T = Row_point;
    size_t size_point = sizeof(float)*Row_point*Col_point;
    point_T_device = DEVICE_MALLOC(point_T_device, size_point);    
    matrix_transpose_cpu(point_device, Row_point, Col_point,
                         point_T_device, Row_point_T, Col_point_T);

    float *K_device = nullptr;
    size_t Row_K = Row_X, Col_K = Col_point_T;
    size_t size_K = sizeof(float)*Row_K*Col_K;
    K_device = DEVICE_MALLOC(K_device, size_K);
    // get K
    gram_cpu(X_device, Row_X, Col_X,
             point_T_device, Row_point_T, Col_point_T,
             K_device, Row_K, Col_K,
             this->param1);

    DEVICE_FREE(point_T_device); 

    //2 - computer K
    //K - mean
    //get mean
    float *mean_device = nullptr;
    size_t Row_mean = 1, Col_mean = Col_K;
    size_t size_mean = sizeof(float)*Row_mean*Col_mean;
    mean_device = DEVICE_MALLOC(mean_device,size_mean);

    // compute mean by row
    // K - repmat(sum(K, 1), [size(K, 1) 1])
    mean_by_rows_cpu(K_device, mean_device, Row_K, Col_K);
    
    float *mean_rep_device = nullptr;
    size_t Row_mean_rep = Row_K, Col_mean_rep = Col_K;
    size_t size_mean_rep = sizeof(float)*Row_mean_rep*Col_mean_rep;
    mean_rep_device = DEVICE_MALLOC(mean_rep_device, size_mean_rep);

    vector_repeat_by_rows_cpu(mean_rep_device, Row_mean_rep, Col_mean_rep,
                              mean_device, Col_mean);

    DEVICE_FREE(mean_device);

    matrix_mul_cpu(K_device, Row_K, Col_K,
                   mean_rep_device, Row_mean_rep, Col_mean_rep,
                   K_device, Row_K, Col_K,
                   SCALAR_TWO_DIVIDE);
    DEVICE_FREE(mean_rep_device);

    // K - J begin
    float *J_T_device = nullptr;
    size_t Row_J_T = Col_K, Col_J_T = Row_K;
    size_t size_J_T = sizeof(float)*Row_J_T*Col_J_T;    
    J_T_device = DEVICE_MALLOC(J_T_device, size_J_T);
    
    vector_repeat_by_rows_cpu(J_T_device, Row_J_T, Col_J_T,
                         (float *)this->column_sums->ptr_buf->ptr_device, this->column_sums->ptr_buf->shape[1] );
    // transpose J
    float *J_device = nullptr;
    size_t Row_J = Col_J_T, Col_J = Row_J_T;
    size_t size_J = size_J_T;
    J_device = DEVICE_MALLOC(J_device, size_J);
    matrix_transpose_cpu(J_T_device, Row_J_T, Col_J_T,
                         J_device, Row_J, Col_J);
    DEVICE_FREE(J_T_device);
    // K-J
    matrix_mul_cpu(K_device, Row_K, Col_K,
                   J_device, Row_J, Col_J,
                   K_device, Row_K, Col_K,
                   SCALAR_TWO_DIVIDE);

    DEVICE_FREE(J_device);

    matrix_add_scalar_cpu(K_device, Row_K, Col_K, this->total_sum);
    
    //3 - transform
    // computer V'*K
    float *t_point2_device = nullptr;   
    size_t Row_t_point2 = this->n_components, Col_t_point2 = Col_K;
    size_t size_t_point2 = sizeof(float)*Row_t_point2*Col_t_point2;
    t_point2_device = DEVICE_MALLOC(t_point2_device, size_t_point2);

    matrix_dotmul_cpu((float *)this->V_T->ptr_buf->ptr_device, this->V_T->ptr_buf->shape[0], this->V_T->ptr_buf->shape[1],
                     K_device, Row_K, Col_K,
                     t_point2_device, Row_t_point2, Col_t_point2); 

    DEVICE_FREE(K_device);
    // copy this->L to L_device
    float *L_device = nullptr;
    size_t Row_L = 1, Col_L = this->n_components;//this->L->ptr_buf->shape[1];
    L_device = DEVICE_MALLOC(L_device, sizeof(float)*Row_L*Col_L);
    matrix_subblock_cpu((float *)this->L->ptr_buf->ptr_device, 1, this->L->ptr_buf->shape[1],
                        L_device, Row_L, Col_L,
                        0,0,Row_L, Col_L);
    // computer invsqrt
    vector_invsqrt_self_cpu(L_device, Col_L);
    
    // compute invsqrtL *ans
    float *invsqrtL_device = nullptr;
    size_t Row_invsqrtL = n_components, Col_invsqrtL = n_components;
    size_t size_invsqrtL = sizeof(float)*Row_invsqrtL*Col_invsqrtL;
    invsqrtL_device = DEVICE_MALLOC(invsqrtL_device, size_invsqrtL);
    matrix_diag_cpu(invsqrtL_device, Row_invsqrtL, Col_invsqrtL,
                     L_device, Col_L);

    DEVICE_FREE(L_device);

    float *res_T_device = nullptr;
    size_t Row_res_T = Row_invsqrtL, Col_res_T = Col_t_point2;
    size_t size_res_T = sizeof(float)*Row_res_T*Col_res_T;
    res_T_device = DEVICE_MALLOC(res_T_device, size_res_T);

    matrix_dotmul_cpu(invsqrtL_device, Row_invsqrtL, Col_invsqrtL,
                      t_point2_device, Row_t_point2, Col_t_point2,
                      res_T_device, Row_res_T, Col_res_T);
    DEVICE_FREE(t_point2_device);
    DEVICE_FREE(invsqrtL_device);

    // computer tpoint= tpoint';
    float *res = (float *)malloc(size_res_T);
    float *res_device = HOST_TO_DEVICE_MALLOC(res, size_res_T);
    size_t Row_res = Col_res_T, Col_res = Row_res_T;
    matrix_transpose_cpu(res_T_device, Row_res_T, Col_res_T,
                         res_device, Row_res, Col_res);
    DEVICE_FREE(res_T_device);
    DEVICE_TO_HOST(res, res_device, size_res_T);

    return new Array{
           nullptr, res, res_device,
           2, {ssize_t(Row_res), ssize_t(Col_res)}, std::string(1,'f'),
           ssize_t(sizeof(float)), ssize_t(Row_res*Col_res),
           {ssize_t(sizeof(float)*Col_res), ssize_t(sizeof(float))}

        };
}


/*
    if ~exist('no_dims', 'var')
        no_dims = 2;
    end
    kernel = 'gauss';
    param1 = 1;
    param2 = 3;
    if nargin > 2
        kernel = varargin{1};
        if length(varargin) > 1 & strcmp(class(varargin{2}), 'double'), param1 = varargin{2}; end
        if length(varargin) > 2 & strcmp(class(varargin{3}), 'double'), param2 = varargin{3}; end
    end

    % Store the number of training and test points
    ell = size(X, 1);

    if size(X, 1) < 2000

        % Compute Gram matrix for training points
        disp('Computing kernel matrix...');
        K = gram(X, X, kernel, param1, param2);

        % Normalize kernel matrix K
        mapping.column_sums = sum(K) / ell;                       % column sums
        mapping.total_sum   = sum(mapping.column_sums) / ell;     % total sum
        J = ones(ell, 1) * mapping.column_sums;                   % column sums (in matrix)
        K = K - J - J';
        K = K + mapping.total_sum;

        % Compute first no_dims eigenvectors and store these in V, store corresponding eigenvalues in L
        disp('Eigenanalysis of kernel matrix...');
        K(isnan(K)) = 0;
        K(isinf(K)) = 0;
        [V, L] = eig(K);
    else
        % Compute column sums (for out-of-sample extension)
        mapping.column_sums = kernel_function([], X', 1, kernel, param1, param2, 'ColumnSums') / ell;
        mapping.total_sum   = sum(mapping.column_sums) / ell;

        % Perform eigenanalysis of kernel matrix without explicitly
        % computing it
        disp('Eigenanalysis of kernel matrix (using slower but memory-conservative implementation)...');
        options.disp = 0;
        options.isreal = 1;
        options.issym = 1;
        [V, L] = eigs(@(v)kernel_function(v, X', 1, kernel, param1, param2, 'Normal'), size(X, 1), no_dims, 'LM', options);
        disp(' ');
    end 

    % Sort eigenvalues and eigenvectors in descending order
    [L, ind] = sort(diag(L), 'descend');
    L = L(1:no_dims);
    V = V(:,ind(1:no_dims));

    % Compute inverse of eigenvalues matrix L
    disp('Computing final embedding...');
    invL = diag(1 ./ L);

    % Compute square root of eigenvalues matrix L
    sqrtL = diag(sqrt(L));

    % Compute inverse of square root of eigenvalues matrix L
    invsqrtL = diag(1 ./ diag(sqrtL));

    % Compute the new embedded points for both K and Ktest-data
    mappedX = sqrtL * V';                     % = invsqrtL * V'* K

    % Set feature vectors in original format
    mappedX = mappedX';

    % Store information for out-of-sample extension
    mapping.X = X;
    mapping.V = V;
    mapping.invsqrtL = invsqrtL;
    mapping.kernel = kernel;
    mapping.param1 = param1;
    mapping.param2 = param2;

 * */
