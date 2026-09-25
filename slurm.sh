#!/bin/bash
#SBATCH --partition=short-28core
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:05:00

module load mpich/gcc/3.2.1

mpicc project1.c -o project
mpirun -np 16 ./project
