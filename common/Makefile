curdir := ${CURDIR}
rootpath := ${curdir}/..

suf := $(shell echo `python3-config --extension-suffix`)
obj := Array$(suf)

CC := g++
NVCC := nvcc
CFLAG := -shared -fPIC -std=c++11
NVCCFLAG := -shared -Xcompiler -fPIC -std=c++11
INCLUDE_PY := $(shell echo `cd ${rootpath}/pybind11 && python3 -m pybind11 --includes`)

INCLUDE_DIR := ${rootpath} ${rootpath}/pybind11/include  
INCLUDE := $(patsubst %,-I%,${INCLUDE_DIR})
INCLUDE += ${INCLUDE_PY}

lib := ${rootpath}/lib

$(info $(INCLUDE_PY))

all: $(obj)
	
$(obj): pybind11.cpp libml4gpu_com_bufex.so libml4gpu_com_mf.so
	$(CC) $< $(CFLAG) $(INCLUDE)  -o ${lib}/$@ -L${lib} -lml4gpu_com_bufex -lml4gpu_com_mf 
	 
libml4gpu_com_bufex.so: buffer_info_ex.cpp libml4gpu_com_mf.so
	$(CC) $< $(CFLAG) $(INCLUDE)  -o ${lib}/$@ -L${lib} -lml4gpu_com_mf

libml4gpu_com_mf.so: malloc_free.cu
	$(NVCC) $< $(NVCCFLAG) ${INCLUDE} -o ${lib}/$@


clean_sotmp := ${obj} libml4gpu_com_mf.so libml4gpu_com_bufex.so
clean_sotmp := $(strip ${clean_sotmp})
comma := ,
empty := 
space := $(empty) $(empty)
clean_so := $(subst $(space),$(comma),${clean_sotmp})

ifneq (1,$(words ${clean_sotmp}))
	clean_so := {$(clean_so)}
endif

clean:
	rm -f ${lib}/${clean_so}
