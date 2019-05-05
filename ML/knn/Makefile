suf := $(shell echo `python3-config --extension-suffix`)
obj = example$(suf)

CC = g++
NVCC = nvcc
CFLAG = -shared -fPIC -std=c++11
NVCCFLAG = -shared -Xcompiler -fPIC -std=c++11

REALPATH = $(realpath .)/../../
INCLUDE_PATH0 = $(REALPATH)pybind11/include $(REALPATH)

INCLUDE_PY = $(shell echo `cd $(REALPATH)pybind11 && python3 -m pybind11 --includes`)

INCLUDE_PATH = $(patsubst %,-I%,$(INCLUDE_PATH0))
INCLUDE_PATH  += $(INCLUDE_PY)

$(info $(INCLUDE_PATH))

all: $(obj)


$(obj): libsort.so
	$(CC) tmp.cpp $(CFLAG) $(INCLUDE_PATH)  -o $@ -L../../common -lmalloc_free



libsort.so:libmalloc_free.so
	$(NVCC) sort.cu

libmalloc_free.so:
	$(NVCC) ../../common/malloc_free.cu $(NVCCFLAG) $(INCLUDE_PATH) -o $@
