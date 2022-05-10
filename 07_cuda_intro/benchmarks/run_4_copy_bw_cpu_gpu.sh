#!/bin/bash

# source code taken from cuda samples, see ${CUDA_ROOT}/samples/1_Utilities/bandwidthTest

cd 4_copy_bw_cpu_gpu

ml purge
ml CUDA GCC/10.2.0

make

./bandwidthTest.x --htod --dtoh

