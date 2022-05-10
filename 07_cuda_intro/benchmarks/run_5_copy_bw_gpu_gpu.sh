#!/bin/bash

# source code taken from OSU Micro-Benchmarks, see https://mvapich.cse.ohio-state.edu/benchmarks/

cd 5_copy_bw_gpu_gpu

ml purge
ml OpenMPI/4.0.6-NVHPC-21.9-CUDA-11.4.1

MY_MPI_ROOT=/apps/all/OpenMPI/4.0.6-NVHPC-21.9-CUDA-11.4.1

if [ ! -f osu_bw.x ]
then
    nvcc -I. -I${MY_MPI_ROOT}/include -D_ENABLE_CUDA_ osu_bw.c osu_util.c osu_util_mpi.c -o osu_bw.x -L${MY_MPI_ROOT}/lib -lmpi -lcuda
fi

mpiexec -n 2 ./osu_bw.x -d cuda -m $((2**10)):$((2**28)) D D
