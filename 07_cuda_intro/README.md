
CUDA introduction
=================





Step 0: Update git repository
-----------------------------

Update the repository on the **Karolina** cluster, go to directory `07_cuda_intro`.





Step 1: Access the Karolina GPU node
------------------------------------

Reservation is set up for the exercise, to access the GPU node, use
```
qsub -q R1078926 -A DD-21-23 -I
```

Later at home, use
```
qsub -q qnvidia -A DD-21-23 -I
```





Step 2: Run the benchmarks
--------------------------

`cd` to the `benchmarks` subdirectory and run the provided benchmark scripts

1. `./run_1_device_query.sh`, scroll through the properties, find the number of GPUs and GPU memory capacity
2. `./run_2_memory_bw_cpu.sh` to measure CPU memory bandwidth
3. `./run_3_memory_bw_gpu.sh` to measure GPU memory bandwidth
4. `./run_4_copy_bw_cpu_gpu.sh` to measure copy bandwidth between CPU and GPU memory
5. `./run_5_copy_bw_gpu_gpu.sh` to measure copy bandwidth between two GPUs

Note the measured values and compare them





Step 3: Hello world in CUDA
---------------------------

### Step 3.1

Write, compile and run a simple Hello world CUDA program. Write a kernel that prints `Hello world`. Launch the kernel with 1 block and 1 thread and wait for its completion. Compile with the `nvcc` compiler.

### Step 3.2

Copy and modify the previous program so that each thread prints its thread index and block index.
Also print the block size and grid size. Launch the kernel with 2 blocks, each with 4 threads.
Try to experiment with different numbers of blocks and threads (kernel configurations).

### Step 3.3

Copy and modify the previous program so that each thread also calculates its global index in the whole grid.


Sample output of the final program (order of the hello messages can differ):
```
Launching the kernel with 2 blocks, each with 4 threads
Kernel was launched, waiting for its completion
Hello from thread 0/4, block 1/2, my global index is 4/8
Hello from thread 1/4, block 1/2, my global index is 5/8
Hello from thread 2/4, block 1/2, my global index is 6/8
Hello from thread 3/4, block 1/2, my global index is 7/8
Hello from thread 0/4, block 0/2, my global index is 0/8
Hello from thread 1/4, block 0/2, my global index is 1/8
Hello from thread 2/4, block 0/2, my global index is 2/8
Hello from thread 3/4, block 0/2, my global index is 3/8
Kernel execution completed
```





Step 4: Vector scale
--------------------

Write a CUDA program that multiplies all elements of a vector by a scalar.

Allocate, initialize and print the vector on the host, allocate device memory and copy the data to the device. 
Write and launch the kernel that performs the scaling, each thread handling at most one element of the vector.
Copy the result back to the host, print it and check for correctness. Use `cudaMalloc`, `cudaMemcpy`, `cudaFree`. Make sure the program works with any vector size.
