#include "common/buffer_info_ex.h"

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

    Array * transform(Array &matrix);

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
