#include <string>
#include "common/include/buffer_info_ex.h"

class KPCA{

    Array *V_T = nullptr;
    Array *L = nullptr;
    Array *column_sums = nullptr; // mean by rows
    float total_sum = 0; //total_sum
    std::string kernel = "gaussian"; // current only support gaussian
    float param1 = 1.0;// gaussian sigma
    size_t n_components = 0;

public:
    KPCA (){}

    // only init the n_components;
    KPCA ( size_t n_components, const std::string kernel_in="gaussian",
         const float param1_in):n_components{n_components},param1{param1_in}{
        kernel = std::move(kernel_in);
        
        V_T = new Array();
        L = new Array();
        column_sums = new Array();
        
    }

    // will stat the matrix, then put the transfer matrix into trans_mat
    Array * fit(Array &matrix);

    Array * transform(Array &train, Array &test);

    ~KPCA(){
        if (column_sums){
            delete column_sums;
            column_sums = nullptr;
        }
        if (V_T){
            delete V_T;
            V_T = nullptr;
        }
        if (L){
            delete L;
            L = nullptr;
        }
    }

};
