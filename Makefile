

common_so:
	$(MAKE) -C common


.PHONY: clean
clean:
	${MAKE} clean -C common 
