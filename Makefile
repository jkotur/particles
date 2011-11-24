
run: cjelly.so
	python jelly.py 

cjelly.so: jelly/jelly.pyx
	cd jelly ; $(MAKE) $(MFLAGS)

