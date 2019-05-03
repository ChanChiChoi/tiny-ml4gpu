
all: example.cpython-36m-x86_64-linux-gnu.so
  g++  -Wall -shared -std=c++11 -fPIC \
        -L.  -lknn  \
        -I/home/zzc/software/pybind11/include \
       `cd /home/zzc/software/pybind11 && python3 -mpybind11 --includes` \
       pybind11.cpp \
       -o  example`python3-config --extension-suffix` 

example.cpython-36m-x86_64-linux-gnu.so: libknn.so
  nvcc --shared -Xcompiler -fPIC cuda_test.cu -o libcutest.so
