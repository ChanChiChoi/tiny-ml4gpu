
all: ml_math ml_pre common_so

ml_math: common_so
	$(MAKE) -C ML/math

ml_pre: common_so
	$(MAKE) -C ML/preprocessing

common_so:
	$(MAKE) -C common


.PHONY: clean
clean:
	${MAKE} clean -C common 
	${MAKE} clean -C ML/math
	${MAKE} clean -C ML/preprocessing
