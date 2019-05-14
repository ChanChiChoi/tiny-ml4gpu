curdir := ${CURDIR}
rootpath := ${curdir}/../..


CC := g++
NVCC := nvcc
CFLAG := -shared -fPIC -std=c++11
NVCCFLAG := -shared -Xcompiler -fPIC -std=c++11
INCLUDE_PY := $(shell echo `cd ${rootpath}/pybind11 && python3 -m pybind11 --includes`)

INCLUDE_DIR := ${rootpath} ${rootpath}/pybind11/include  
INCLUDE := $(patsubst %,-I%,${INCLUDE_DIR})
INCLUDE += ${INCLUDE_PY}

libinc := ${rootpath}/lib
libres := ${rootpath}/ML/lib

obj := libml4gpu_ml_preprocess.so

all: $(obj)
	
$(obj):  data.cu
	$(NVCC) $^ $(NVCCFLAG) $(INCLUDE)  -o ${libres}/$@ -L${libinc} -lmalloc_free -lbuffer_info_ex 
	 

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

clean:
	-rm -f ${libres}/${clean_so}
