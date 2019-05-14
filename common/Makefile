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
	
$(obj): pybind11.cpp libbuffer_info_ex.so libmalloc_free.so
	$(CC) $< $(CFLAG) $(INCLUDE)  -o ${lib}/$@ -L${lib} -lmalloc_free -lbuffer_info_ex 
	 
libbuffer_info_ex.so: buffer_info_ex.cpp libmalloc_free.so
	$(CC) $< $(CFLAG) $(INCLUDE)  -o ${lib}/$@ -L${lib} -lmalloc_free

libmalloc_free.so: malloc_free.cu
	$(NVCC) $< $(NVCCFLAG) ${INCLUDE} -o ${lib}/$@


clean_sotmp := ${obj} libmalloc_free.so libbuffer_info_ex.so
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
