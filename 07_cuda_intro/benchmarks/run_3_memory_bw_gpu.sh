#!/bin/bash

# this is BabelStream benchmark, see http://uob-hpc.github.io/BabelStream/

cd 3_memory_bw_gpu

ml purge
ml CUDA GCC/10.2.0

make -f CUDA.make

./cuda-stream.x -s $((2**26))
