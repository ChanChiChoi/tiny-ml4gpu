curdir := ${CURDIR}
rootpath := ${curdir}/../../..
mlpath := ${rootpath}/ml

suf := $(shell echo `python3-config --extension-suffix`)

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
libtop := $(rootpath)/lib

LDFLAGS_DIR := ${libcommon} ${libml} $(libtop)
LDFLAGS := $(patsubst %,-L%,${LDFLAGS_DIR})

LDLIBS_DIR := m4g_com_cuMF m4g_ml_math m4g_ml_preprocess
LDLIBS := $(patsubst %,-l%,${LDLIBS_DIR})

pcaobj := PCA${suf}
kpcaobj := KPCA$(suf)
all:  $(pcaobj) $(kpcaobj)
	

${pcaobj}: pybind11.cpp pca.o matrix.o
	cp ${libml}/lib/matrix.o .
	$(CC) $^ ${libcommon}/buffer_info_ex.o ${CLDFLAG} ${INCLUDE} -o $@ ${LDFLAGS} ${LDLIBS}

pca.o:  pca.cpp 
	$(CC) $^ $(CXXFLAG) $(INCLUDE)  -o $@ 
	 
${kpcaobj}: pybind11_kernel_pca.cpp kernel_pca.o
	$(CC) $^ ${libcommon}/buffer_info_ex.o ${CLDFLAG} ${INCLUDE} -o $@ ${LDFLAGS} ${LDLIBS}

kernel_pca.o: kernel_pca.cpp
	$(CC) $^ $(CXXFLAG) $(INCLUDE)  -o $@

clean_sotmp := ${pcaobj} ${kpcaobj}
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

clean:
	rm -f *.o
	rm -f ${clean_so}
	-rm -f ${libml}/${clean_so}
