
all:  ml_mk

ml_mk: common_install
	$(MAKE) -C ml

common_install:common_mk
	$(MAKE) install -C common

common_mk:
	$(MAKE) -C common

.PHONY: install
install:mv_lib

mv_lib: ml_install
	mv common/lib/*.so lib
	mv ml/lib/*.so lib

ml_install:ml_mk
	${MAKE} install -C ml

.PHONY: clean
clean:
	${MAKE} clean -C common 
	${MAKE} clean -C ml
	rm -f lib/*
