#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "common/malloc_free.h"
#include "common/helper.cuh"

template<typename T> void
_svd(T *A_device, const int Row_A, const int Col_A, const int lda,
    T *U_device, const int Row_U, const int Col_U,
    T *S_device, const int Length,
    T *VT_device, const int Row_VT, const int Col_VT){

  cusolverDnHandle_t cusolverH = NULL;
  cublasHandle_t cublasH = NULL;

  T *rwork_device = NULL;

  int lwork = 0;
  int info_gpu = 0;

  //step1:
  CHECK_CALL_DEFAULT(cusolverDnCreate(&cusolverH));
  CHECK_CALL_DEFAULT(cublasCreate(&cublasH));

  //step2:
//  T *A_device = HOST_TO_DEVICE_MALLOC(A, sizeof(T)*lda*Col_A)；
//  T *S_device = DEVICE_MALLOC(sizeof(T)*Col_A)；
//  T *U_device = DEVICE_MALLOC(sizeof(T)*lda*Row_A)；
//  T *VT_device = DEVICE_MALLOC(sizeof(T)*lda*Col_A)；
  
  int *devInfo_device;
  DEVICE_MALLOC(devInfo_device,sizeof(int));

  //step3
  CHECK_CALL_DEFAULT(cusolverDnDgesvd_bufferSize(cusolverH,Row_A,Col_A,&lwork));

  T *work_device;
  work_device = DEVICE_MALLOC(work_device,sizeof(T)*lwork);

  //step 4:compute svd
  signed char jobu = 'A';
  signed char jobvt = 'A';
  CHECK_CALL_DEFAULT(cusolverDnDgesvd(cusolverH, jobu, jobvt, 
                          Row_A, Col_A,
                          A_device, lda,
                          S_device, U_device,lda, 
                          VT_device, lda, work_device, lwork, rwork_device, devInfo_device));

  CHECK_CALL_DEFAULT(cudaDeviceSynchronize());
 
  DEVICE_TO_HOST_FREE(&info_gpu, devInfo_device, sizeof(int));  

  DEVICE_FREE(work_device);
  DEVICE_FREE(rwork_device);

  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);

  cudaDeviceReset();
  return ;
}


void
svd(float *A, const int Row_A, const int Col_A, const int lda,
    float *U, const int Row_U, const int Col_U,
    float *S, const int Length,
    float *VT, const int Row_VT, const int Col_VT){

    _svd<float>(A, Row_A, Col_A, lda,
        U, Row_U, Col_U,
        S, Length,
        VT, Row_VT, Col_VT);
}


void
svd(double *A, const int Row_A, const int Col_A, const int lda,
    double *U, const int Row_U, const int Col_U,
    double *S, const int Length,
    double *VT, const int Row_VT, const int Col_VT){

    _svd<double>(A, Row_A, Col_A, lda,
        U, Row_U, Col_U,
        S, Length,
        VT, Row_VT, Col_VT);
}



