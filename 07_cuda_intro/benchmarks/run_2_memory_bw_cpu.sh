#!/bin/bash

# this is the STREAM benchmark, see https://www.cs.virginia.edu/stream/

cd 2_memory_bw_cpu

ml purge
ml GCC/10.2.0

export GOMP_CPU_AFFINITY=0-64:8
export OMP_NUM_THREADS=8

if [ ! -f stream.x ]
then
    gcc -fopenmp -O3 -DSTREAM_ARRAY_SIZE=250000000 -mcmodel=medium stream.c -o stream.x
fi

./stream.x
