
nvcc operation is different with gcc/g++, so I encounter some weird problems. that is frustrating.


#### 1. undefined symbol
if you was reported "undefined symbol", when you "nm" the .so file or .o file, that the symbol exists. what confused me is the problem do not always happen.

math.h
```
#pragma once
void display();
```

math.cu
```
#include <stdio.h>
#include "math.h"

void
display(int a){
 printf("hello\n");
}
```

```
nvcc math.cu -c -shared -Xcompiler -fPIC
nvcc math.o -shared -Xcompiler -fPIC -o libmath.so
```
test.cpp
```
#include<stdio.h>
#include "math.h"

int
sofile(){
 display();
 return 0;
}
```
then, 
```
g++ test.cpp -L. -lmath -shared -fPIC -o libtest.so
```
if you "ldd -r libtest.so", it will report "undefined symbol: _Z7displayv   (./libtest.so)";  
but when "nm libmath.so |grep display", it says "000000000000626c T _Z7displayi".

you should .....



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





