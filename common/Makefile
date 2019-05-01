
suf := $(shell echo `python3-config --extension-suffix`)
obj = Array$(suf)

CC = g++
NVCC = nvcc
CFLAG = -shared -fPIC -std=c++11
NVCCFLAG = -shared -Xcompiler -fPIC -std=c++11
INCLUDE_PY = $(shell echo `cd ../pybind11 && python3 -m pybind11 --includes`)

$(info $(INCLUDE_PY))

all: $(obj)
	

$(obj): libbuffer_info_ex.so
	$(CC) pybind11.cpp $(CFLAG) $(INCLUDE_PY) -I../ -I../pybind11/include -o $@ -L. -lmalloc_free -lbuffer_info_ex 

	 

libbuffer_info_ex.so:libmalloc_free.so
	$(CC) buffer_info_ex.cpp $(CFLAG) $(INCLUDE_PY) -I../ -I../pybind11/include -o $@ -L. -lmalloc_free

libmalloc_free.so:
	$(NVCC) malloc_free.cu $(NVCCFLAG) -I../ -o $@
