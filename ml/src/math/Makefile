curdir := ${CURDIR}
rootpath := ${curdir}/../../..
mlpath := ${rootpath}/ml

CC := g++
NVCC := nvcc
CXXFLAG := -c -fPIC -std=c++11
CLDFLAG := -shared -fPIC -std=c++11
NVCCFLAG := -shared -Xcompiler -fPIC -std=c++11
INCLUDE_PY := $(shell echo `cd ${rootpath}/pybind11 && python3 -m pybind11 --includes`)

INCLUDE_DIR := ${rootpath} ${rootpath}/pybind11/include  
INCLUDE_SELF := -I${rootpath} -I/usr/local/cuda/include
INCLUDE := $(patsubst %,-I%,${INCLUDE_DIR})
INCLUDE += ${INCLUDE_PY}


libcommon := ${rootpath}/common/lib
libml := ${rootpath}/ml/lib

LDFLAGS_DIR := /usr/local/cuda/lib64
LDFLAGS := $(patsubst %,-L%,${LDFLAGS_DIR})

LDFLAGS_EX_DIR := ${libcommon} $(libml)
LDFLAGS_EX := $(patsubst %,-L%,${LDFLAGS_EX_DIR})

LDLIBS_ITEMS := cudart cublas cusolver gomp
LDLIBS := $(patsubst %,-l%,${LDLIBS_ITEMS})

LDLIBS_EX_ITEMS := m4g_com_cuMF m4g_ml_math
LDLIBS_EX := $(patsubst %,-l%,${LDLIBS_EX_ITEMS})

obj := libm4g_ml_math.so

all: $(obj)
	
$(obj):  matrix.o svd.o stats.o vector.o
	$(CC)  $^ $(CLDFLAG) $(INCLUDE_SELF)  -o $@ ${LDFLAGS} ${LDLIBS}
	 
matrix.o: matrix.cu
	$(NVCC)  $^  -c $(NVCCFLAG) $(INCLUDE_SELF)  -o $@

svd.o:svd.cu
	$(NVCC) $^ -c $(NVCCFLAG) $(INCLUDE_SELF)  -o $@

stats.o: stats.cpp
	$(CC) $^ -c $(CXXFLAG) $(INCLUDE_SELF) -o $@ 

vector.o: vector.cu
	$(NVCC) $^ -c $(NVCCFLAG) $(INCLUDE_SELF) -o $@

clean_sotmp := ${obj} 
clean_sotmp := $(strip $(clean_sotmp))
comma := ,
empty := 
space := $(empty) $(empty)
clean_so := $(subst $(space),$(comma),${clean_sotmp})

ifneq (1,$(words ${clean_sotmp}))
	clean_so := {$(clean_so)}
endif

$(info ${clean_so})

install:
	cp *.so ${libml}

test: test_svd

test_svd:test_svd.cu
	$(NVCC) $^  $(INCLUDE_SELF) $(LDFLAGS) $(LDFLAGS_EX) $(LDLIBS) $(LDLIBS_EX) -o $@
	CUDA_VISIBLE_DEVICES='1' LD_LIBRARY_PATH=$(libcommon):$(libml):$(LD_LIBRARY_PATH) ./$@



clean:
	rm -f *.o
	rm -f ${clean_so}
	-rm -f ${libml}/${clean_so}
	rm -f test_svd
