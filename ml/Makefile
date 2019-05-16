
all: models


models: decomposition_mk
	

decomposition_mk:math_install prepro_install
	$(MAKE) -C src/decomposition

math_install:math_mk
	$(MAKE) install -C src/math

prepro_install:prepro_mk
	$(MAKE) install -C src/preprocessing

math_mk:
	$(MAKE) -C src/math

prepro_mk:
	$(MAKE) -C src/preprocessing


install:
	$(MAKE) install -C src/decomposition

clean:
	${MAKE} clean -C src/math
	${MAKE} clean -C src/preprocessing
	$(MAKE) clean -C src/decomposition
