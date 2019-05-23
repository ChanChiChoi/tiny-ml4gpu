#include <string>
#include "common/include/buffer_info_ex.h"

class KPCA{

    Array *V = nullptr;
    Array *invsqrtL = nullptr;
    Array *column_sums = nullptr; // mean by rows
    float *total_sum = 0; //total_sum
    std::string kernel = "gaussian"; // current only support gaussian
    float *param1 = 1.0;// gaussian sigma
    size_t n_components = 0;

public:
    KPCA (){}

    // only init the n_components;
    KPCA ( size_t n_components, const std::string kernel_in="gaussian"):n_components{n_components}{
        kernel = std::move(kernel_in);
        trans_mat = new Array();
        mean_vec = new Array();
    }

    // will stat the matrix, then put the transfer matrix into trans_mat
    Array * fit(Array &matrix);

    Array * transform(Array &train, Array &test);

    ~KPCA(){
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
