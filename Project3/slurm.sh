#!/bin/bash
#SBATCH --output=project_3_problem_3.1_strassen_1_serial_256n_%j.out
#SBATCH --partition=short-96core-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:02:00

module load mpich/gcc/3.4.2

#mpicc project_2_problem_2.4.c -o project_2 -lm


mpicc -O3 -std=c11 mpi_strassen.c -lm -o strassen_mm


export UCX_TLS=tcp,self

#mpirun -np 4 ./mpi_fox 256
mpirun -np 1 ./strassen_mm 256 1


