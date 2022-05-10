
Matrices
========





Task 0: Update git repository, access Karolina GPU node
-------------------------------------------------------

Update the repository on the **Karolina** cluster, go to directory `12_matrices`.

Reservation is set up for the exercise, to access the GPU node, use
```
qsub -q R1078931 -A DD-21-23 -I
```

Later at home, use e.g.
```
qsub -q qnvidia -A DD-21-23 -l walltime=4:00:00 -I
```

Load the CUDA module
```
ml CUDA
```





Task 1: Nsight profiling
------------------------

Lecture and exercise on gpu profiling by Radim Vavřík. Work in the `gpu_profiling` directory.





Task 2: Matrix transpose
------------------------

Work with the `transpose.cu` file. All memory management, along with kernel launches is done for you. You only need to implement the CUDA kernels. We use pitched memory, so you will need to use `ld` in the matrix indexing.

### Task 2.1: Naive implementation

Implement a basic matrix transpose kernel performing a transposition of a square matrix. Measure its execution time and compare it with the copy kernel. Why is the performance of transposition lower, compared to the copy?

### Task 2.2: Shared memory

Resolve the problem of uncoalesced global memory accesses using tiling shared memory. Load a whole tile into the shared memory, and then store it transposed. Compare the execution time. Why does the execution time still not match the copy kernel?

### Task 2.3: Bank conflicts

Resolve the shared memory bank conflicts by using a simple trick presented in one of the previous lectures.





Task 3: Matrix-matrix multiplication
------------------------------------

Work with the `mat_mul.cu` file. Again, all memory management and kernel launches is done for you, you just need to implement the kernels. Matrices are stored in row-major order, and we use pitched memory.

### Task 3.1: Naive implementation

Implement a simple matrix-matrix multiplication kernel, performing the operation `C = A * B`, where all the matrices are square. Two-dimensional kernel is used, each thread should calculate one element of the output matrix. Measure the kernel execution time.

### Task 3.2 Tiled matrix multiplication

Use shared memory and tiling to improve the performance of the matrix multiplication kernel. Measure the kernel execution time and compare it to the previous version.
