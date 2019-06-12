#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "common/include/malloc_free.h"
#include "common/include/helper.cuh"
#include "ml/include/math/svd.h"


void
svd(float *A_device, const int Row_A, const int Col_A, const int lda,
    float *U_device, const int Row_U, const int Col_U,
    float *S_device, const int Length,
    float *VT_device, const int Row_VT, const int Col_VT){

  assert(Row_A >= Col_A);
  cusolverDnHandle_t cusolverH = NULL;
  cublasHandle_t cublasH = NULL;
  float *rwork_device = NULL;

  int lwork = 0;
  int info_gpu = 0;

  CHECK_CALL_DEFAULT(cusolverDnCreate(&cusolverH));
  CHECK_CALL_DEFAULT(cublasCreate(&cublasH));

  int *devInfo_device;
  devInfo_device = DEVICE_MALLOC(devInfo_device,sizeof(int));

  //cusolverDnSgesvd_bufferSize single precision
  CHECK_CALL_DEFAULT(cusolverDnSgesvd_bufferSize(cusolverH,Row_A,Col_A,&lwork));

  float *work_device;
  work_device = DEVICE_MALLOC(work_device,sizeof(float)*lwork);

  //step 4:compute svd
  signed char jobu = 'A';
  signed char jobvt = 'A';
  //cusolverDnSgesvd single precision
  CHECK_CALL_DEFAULT(cusolverDnSgesvd(cusolverH, jobu, jobvt, 
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
    
  return ;
}

void
svd(double *A_device, const int Row_A, const int Col_A, const int lda,
    double *U_device, const int Row_U, const int Col_U,
    double *S_device, const int Length,
    double *VT_device, const int Row_VT, const int Col_VT){

  assert(Row_A >= Col_A);
  cusolverDnHandle_t cusolverH = NULL;
  cublasHandle_t cublasH = NULL;
  double *rwork_device = NULL;

  int lwork = 0;
  int info_gpu = 0;

  CHECK_CALL_DEFAULT(cusolverDnCreate(&cusolverH));
  CHECK_CALL_DEFAULT(cublasCreate(&cublasH));

  int *devInfo_device;
  devInfo_device = DEVICE_MALLOC(devInfo_device,sizeof(int));

  //cusolverDnDgesvd_bufferSize double precision
  CHECK_CALL_DEFAULT(cusolverDnDgesvd_bufferSize(cusolverH,Row_A,Col_A,&lwork));

  double *work_device;
  work_device = DEVICE_MALLOC(work_device,sizeof(double)*lwork);

  //step 4:compute svd
  signed char jobu = 'A';
  signed char jobvt = 'A';
  //cusolverDnDgesvd double precision
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
    
  return ;
}
