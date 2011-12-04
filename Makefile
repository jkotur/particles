
run: cjelly.so jelly_kernel.cubin
	python jelly.py 

cjelly.so: jelly/jelly.pyx
	cd jelly ; $(MAKE) $(MFLAGS)

jelly_kernel.cubin: jelly_kernel.cu
	nvcc --cubin -arch sm_11 -I/usr/include/pycuda jelly_kernel.cu

clean:
	rm jelly_kernel.cubin cjelly.so
