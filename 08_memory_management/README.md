
Memory management
=================





Task 0: Update git repository, access Karolina GPU node
-------------------------------------------------------

Update the repository on the **Karolina** cluster, go to directory `08_memory_management`.

Reservation is set up for the exercise, to access the GPU node, use
```
qsub -q R1078927 -A DD-21-23 -I
```

Later at home, use e.g.
```
qsub -q qnvidia -A DD-21-23 -l walltime=4:00:00 -I
```

Load the CUDA module
```
ml CUDA
```





Task 1: Error checking in CUDA
------------------------------

Modify the `vscale_cudaerror.cu` file containing a vector scale program. Check for errors after every CUDA function call. Write a macro to do this. Use `cudaError_t`, `cudaGetErrorString()`. Try to create an error to test it.





Task 2: Vector add
------------------

### Task 2.1: Explicit memory transfers

Write a program that generates two vectors on the host, adds these vectors into a third vector on the GPU device, performing the operation `c = a + b`, then check the result for correctness on the host. Optionally print the three vectors. Use explicit memory transfers and check for errors. Use the `vadd_1_base.cu` file as a base.

Sample output:
```
Input A:
  0.000   1.000   2.000   3.000   4.000   5.000   6.000   7.000   8.000   9.000  10.000
Input B:
  0.000  10.000  20.000  30.000  40.000  50.000  60.000  70.000  80.000  90.000 100.000
Output C:
  0.000  11.000  22.000  33.000  44.000  55.000  66.000  77.000  88.000  99.000 110.000
The result is CORRECT!
```

### Task 2.2: Pinned host memory

Copy and modify the previous program such that it uses pinned page-locked memory allocations on the host to increase data transfer bandwidth. Use `cudaMallocHost()` or `cudaHostAlloc()`, and `cudaFreeHost()`.

### Task 2.3: Managed memory

Copy and modify the previous program such that it uses managed (unified) memory. Use `cudaMallocManaged()`.

Bonus: use `cudaMemPrefetch()` to prefetch the vectors data to the device.

### Task 2.4: Utilize multiple GPUs

Copy the previous program and modify it such that it utilizes all available GPUs. Split the vectors into as many segments as there are devices in the system - calculate start indexes and sizes of the segments, do not issue any extra allocations. Use `cudaGetDeviceCount()` and `cudaSetDevice()`. Use a for loop to iterate through the devices.

Think about how you could use OpenMP or MPI to submit kernels to multiple GPUs in parallel.

Experiment with `CUDA_VISIBLE_DEVICES` environment variable.

### Task 2.5: Threads to elements mapping

Copy and modify any of the previous programs such that the kernel is launched with a fixed number of threads and blocks. You will have to loop through all the elements inside the kernel.

Bonus: launch the kernel with twice as many blocks as there are streaming multiprocessors on the device (use `cudaDeviceGetAttribute()` or `cudaGetDeviceProperties()`).

### Task 2.6: Multiple GPUs with explicit memory transfers

Think about what would have to be done so that the explicit memory transfer version of the vector add program would utilize multiple GPUs.





Task 3: Matrices, multi-dimensional kernels
-------------------------------------------

### Task 3.1: Matrix scale

Implement a matrix scale operation (`A = c * A`) in CUDA using two-dimensional grid and blocks. Use row-major order to store the matrix in memory. Allocate separate buffers in host and device memory and transfer the data explicitly using `cudaMemcpy()`. Make sure the program works with any sensible matrix size. Mind the maximum number of threads per block, 1024. Use the `mscale_1_base.cu` file as a base.

### Task 3.2: Pitched memory

Copy and modify the previous program such that it uses `cudaMallocPitch()` for device memory allocation. Modify the kernel and other code accordingly. Use `cudaMemcpy2D()` for host <-> device data transfers. Think about the difference between pitch, leading dimension and width.
