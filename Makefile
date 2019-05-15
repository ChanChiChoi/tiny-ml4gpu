
all: common_mk ml_mk

ml_mk: common_mk
	$(MAKE) -C ml

common_mk:
	$(MAKE) -C common


.PHONY: install
install:
	${MAKE} install -C common
	${MAKE} clean -C ml

.PHONY: clean
clean:
	${MAKE} clean -C common 
	${MAKE} clean -C ml
