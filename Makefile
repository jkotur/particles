
run: water_kernel.cubin
	python particles.py 

water_kernel.cubin: water_kernel.cu
	nvcc --cubin -arch sm_11 -I/usr/include/pycuda water_kernel.cu

clean:
	rm water_kernel.cubin *pyc
