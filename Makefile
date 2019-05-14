
all: ml_math common_so

ml_math: common_so
	$(MAKE) -C ML/math

common_so:
	$(MAKE) -C common


.PHONY: clean
clean:
	${MAKE} clean -C common 
	${MAKE} clean -C ML/math
