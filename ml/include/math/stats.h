#pragma once
#include "common/include/type.h"

void
cov_cpu(float *mat, u32 Row_mat, u32 Col_mat,
        float *mat_cov, u32 Row_mat_cov, u32 Col_mat_cov);


void
gram_cpu(float *mat, u32 Row_mat, u32 Col_mat,
         float *mat_gram, u32 Row_gram, u32 Col_gram );
