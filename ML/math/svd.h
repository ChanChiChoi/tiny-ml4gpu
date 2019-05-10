#pragma once

void
svd(float *A, const int Row_A, const int Col_A, const int lda,
    float *U, const int Row_U, const int Col_U,
    float *S, const int Length,
    float *VT, const int Row_VT, const int Col_VT);


void
svd(double *A, const int Row_A, const int Col_A, const int lda,
    double *U, const int Row_U, const int Col_U,
    double *S, const int Length,
    double *VT, const int Row_VT, const int Col_VT);
