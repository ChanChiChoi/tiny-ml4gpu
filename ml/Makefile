
all: src_mk


src_mk:
	$(MAKE) -C src/math
	$(MAKE) -C src/preprocessing

install:
	$(MAKE) install -C src/math
	$(MAKE) install -C src/preprocessing

clean:
	${MAKE} clean -C src/math
	${MAKE} clean -C src/preprocessing
