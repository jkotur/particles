
run: sph_kernel.cubin lbm_kernel.cubin
	python particles.py 

sph_kernel.cubin: sph_kernel.cu
	nvcc --cubin -arch sm_11 -I/usr/include/pycuda sph_kernel.cu

lbm_kernel.cubin: lbm_kernel.cu
	nvcc --cubin -arch sm_11 -I/usr/include/pycuda lbm_kernel.cu

clean:
	rm sph_kernel.cubin lbm_kernel.cubin *pyc

