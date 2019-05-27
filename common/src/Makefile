curdir := ${CURDIR}
rootpath := ${curdir}/../..
commonpath := ${curdir}/..

suf := $(shell echo `python3-config --extension-suffix`)
obj := Array$(suf)

CC := g++
NVCC := nvcc
CXXFLAG := -c  -fPIC -std=c++11
CLDFLAG :=  -shared -fPIC -std=c++11
NVCCFLAG := -shared -Xcompiler -fPIC -std=c++11
INCLUDE_PY := $(shell echo `cd ${rootpath}/pybind11 && python3 -m pybind11 --includes`)

INCLUDE_SELF := -I${rootpath}

INCLUDE_DIR := ${rootpath} ${rootpath}/pybind11/include  
INCLUDE := $(patsubst %,-I%,${INCLUDE_DIR})
INCLUDE += ${INCLUDE_PY}

libcom := ${commonpath}/lib

$(info $(INCLUDE_PY))

all:  libm4g_com_cuMF.so ${obj}
	
# get the final pybind11 .so file
$(obj): buffer_info_ex.o pybind11.cpp
	$(CC) $^ ${CLDFLAG} $(INCLUDE)  -o $@ -L. -lm4g_com_cuMF 
	 
# use -fPIC to compiler the cpp file 
buffer_info_ex.o: buffer_info_ex.cpp
	$(CC) $< $(CXXFLAG) $(INCLUDE)  -o $@ 

# make cu tobe a .so file
libm4g_com_cuMF.so: malloc_free.cu
	$(NVCC) $< $(NVCCFLAG) ${INCLUDE_SELF} -o $@


clean_sotmp := ${obj} libm4g_com_cuMF.so 
clean_sotmp := $(strip ${clean_sotmp})
comma := ,
empty := 
space := $(empty) $(empty)
clean_so := $(subst $(space),$(comma),${clean_sotmp})

ifneq (1,$(words ${clean_sotmp}))
	clean_so := {$(clean_so)}
endif


install:
	cp *.so ${libcom}
	cp *.o ${libcom}

clean:
	rm -f *.o
	rm -f ${clean_so}
	rm -f ${libcom}/${clean_so}
