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
INCLUDE_SELF := -I${rootpath}
INCLUDE := $(patsubst %,-I%,${INCLUDE_DIR})
INCLUDE += ${INCLUDE_PY}

libcommon := ${rootpath}/common/lib
libml := ${rootpath}/ml/lib

LDFLAGS_DIR := /usr/local/cuda/lib64
LDFLAGS := $(patsubst %,-I%,${LDFLAGS_DIR})

LDLIBS_ITEMS := cudart cublas cusolver
LDLIBS := $(patsubst %,-l%,${LDLIBS_ITEMS})

obj := libm4g_ml_math.so

all: $(obj)
	
$(obj):  matrix.o svd.o 
	$(NVCC) $^ $(NVCCFLAG) $(INCLUDE_SELF)  -o $@ ${LDFLAGS} ${LDLIBS}
	 
matrix.o:matrix.cu
	$(NVCC) $^ $(NVCCFLAG) $(INCLUDE_SELF)  -o $@

svd.o:svd.cu
	$(NVCC) $^ $(NVCCFLAG) $(INCLUDE_SELF)  -o $@

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
	mv *.so ${libml}

clean:
	rm -f *.o
	rm -f ${clean_so}
	-rm -f ${libml}/${clean_so}
