#include <stdio.h>
#include "ml/include/math/svd.h"
#include "common/include/malloc_free.h"



void printMatrix(int m, int n, const double*A, int lda, const char* name){

  for(int row = 0 ; row < m ; row++){
    for(int col = 0 ; col < n ; col++){
        double Areg = A[row + col*lda];
        printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
    }
  }
}

int
main(){
    
  const int m = 1000;
  const int n = 60;
  const int lda = m;

  double *A = (double *)malloc(sizeof(double)*m*n);
  for (int i=0;i<m*n;i++){
    A[i] = 3.0;

  }
//  double A[lda*n] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
  double U[lda*m]; // m-by-m unitary matrix
  double VT[n*n]; // n-by-n unitary matrix
  double S[n]; // singular value
//  double S_exact[n] = {7.065283497082729, 1.040081297712078};

  double *A_device = HOST_TO_DEVICE_MALLOC(A,sizeof(double)*lda*n);
  double *U_device = HOST_TO_DEVICE_MALLOC(U,sizeof(double)*lda*m);
  double *VT_device = HOST_TO_DEVICE_MALLOC(VT,sizeof(double)*lda*n);
  double *S_device = HOST_TO_DEVICE_MALLOC(S,sizeof(double)*n);

  const int Row_A = m;
  const int Col_A = n;
  const int Row_U = lda;
  const int Col_U = n;
  const int Length = n;
  const int Row_VT = n;
  const int Col_VT = n;

  svd(A_device, Row_A, Col_A, lda,
    U_device, Row_U, Col_U,
    S_device, Length,
    VT_device, Row_VT, Col_VT);

   DEVICE_TO_HOST_FREE(A,A_device, sizeof(double)*lda*n);
   DEVICE_TO_HOST_FREE(U,U_device, sizeof(double)*lda*m);
   DEVICE_TO_HOST_FREE(VT,VT_device, sizeof(double)*lda*n);
   DEVICE_TO_HOST_FREE(S,S_device, sizeof(double)*n);

   

printf("S = (matlab base-1)\n");
printMatrix(n, 1, S, lda, "S");
printf("=====\n");
//
//printf("U = (matlab base-1)\n");
//printMatrix(m, m, U, lda, "U");
//printf("=====\n");
//
//printf("VT = (matlab base-1)\n");
//printMatrix(n, n, VT, lda, "VT");
//printf("=====\n");


 return 0;


}
