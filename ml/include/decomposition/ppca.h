#include "common/include/buffer_info_ex.h"

class PPCA{

    Array *trans_mat = NULL;
    Array *mean_vec = NULL;
    size_t min_obs = 0;
    float tol = 1e-4;
    

public:
    PPCA (){}

    // only init the n_components;
    PPCA (float tol, size_t min_obs):tol{tol},n_components{n_components}{
        trans_mat = new Array();
        mean_vec = new Array();
    }

    // will stat the matrix, then put the transfer matrix into trans_mat
    PPCA & fit(Array &matrix);

    Array * transform(Array &matrix);

    ~PPCA(){
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
