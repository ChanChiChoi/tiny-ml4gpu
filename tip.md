
nvcc operation is different with gcc/g++, so I encounter some weird problems. that is frustrating.


#### 1. undefined symbol
I have a .cu file
```
void text(){
  /*some coding*/
}
```
then, 




#### 2. not create .so with .cu and .cpp

do not create one .so file with some .cu files and some .cpp files, which will result some problems that you can not understand, for example, 

matrix.cu
```
matrix_mul(){
 /*some coding*/
}
```

stats.cpp
```
#include "matrix.h"
cov_cpu(){
 matrix_mul();
}
```

create libmath.so(matrix.o, stats.o)

pca.cpp
```
cov_cpu()
cudaMalloc(); // or cudaMemcpy or some others
```
It will report cudaMalloc error(cuda IllAddress), or some others reason(helpless), the first line which below "conv_cpu" and access device memory. so we must rename stats.cpp to stats.cu, then "nvcc"(do not use gcc/g++) all the files inside this .so file.





