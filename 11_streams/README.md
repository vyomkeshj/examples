
Streams
=======





Task 0: Update git repository, access Karolina GPU node
-------------------------------------------------------

Update the repository on the **Karolina** cluster, go to directory `11_streams`.

Reservation is set up for the exercise, to access the GPU node, use
```
qsub -q R1078930 -A DD-21-23 -I
```

Later at home, use e.g.
```
qsub -q qnvidia -A DD-21-23 -l walltime=4:00:00 -I
```

Load the CUDA module
```
ml CUDA
```





Task 1: Histogram
-----------------

### Task 1.1

In the file `histogram.cu`, implement and launch a kernel computing a histogram -- count how many times does each number in $\langle 0,100 )$ appear in the input array. Use atomic operations,
mainly the `atomicAdd()` function. Measure the execution time of the kernel (you can use the provided `CUDAMEASURE` macro).

### Task 1.2

Improve the performance of the histogram kernel by using dynamic shared memory. Compute a histogram block-wide histo. Measure the execution time.

### Task 1.3

Launch the kernel with 432 blocks, each with 1024 threads (432 = 4\*108, there are 108 streaming multiprocessors on the A100 GPU). There are not enough threads for a one-to-one thread-element mapping, the threads will need to iterate through the array. Measure the execution time. Why do you think the performance improved?

Bonus: find out the number of streaming multiprocessors at runtime using `cudaDeviceGetAttribute()`. Launch the kernel with 4 times as much blocks.


Task 2: Asynchronous copies, streams
------------------------------------

Let us return to the vector add example. We will modify it to use streams. Start from the `vadd.cu` file in this directory, where the basic vector add example is completely implemented. In each subtask, copy and modify the code enclosed in `{ }`. In each subtasks, think about in what order and in what streams will the individual operations execute.

### Task 2.1

Use asynchronous copying (`cudaMemcpyAsync()`) instead of the blocking variant `cudaMemcpy()` we were using until now. Do not forget that `cudaMemcpy()` was blocking, but `cudaMemcpyAsync()` is not.

### Task 2.2

Create a stream (use `cudaStream_t`, `cudaStreamCreate()`, `cudaStreamDestroy()`) and submit all the operations (copyin, kernel, copyout) into the created stream.

### Task 2.3

Split the whole vector add operation into multiple smaller vector adds. Split the vector into 50 segments, then in a loop copy each of the segment's data, add the vector segments, and copy the data back. The copies will again be asynchronous and all operations submitted into the created stream.

### Task 2.4

Create 4 streams in a `std::vector`. Submit each segment's work into a different stream, reusing streams from the beginning if necessary.
