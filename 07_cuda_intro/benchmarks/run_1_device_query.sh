#!/bin/bash

# source code taken from cuda samples, see ${CUDA_ROOT}/samples/1_Utilities/deviceQuery

ml purge
ml CUDA GCC/10.2.0

unset CUDA_VISIBLE_DEVICES

cd 1_device_query

make

./deviceQuery.x
