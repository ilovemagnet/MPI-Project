#!/bin/bash
#SBATCH --output=project_2_problem_2.4_4p_%j.out
#SBATCH --partition=hbm-1tb-long-96core
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:05:00

module load mpich/gcc/3.4.2

mpicc project_2_problem_2.4.c -o project_2 -lm

export UCX_TLS=tcp,self

mpirun -np 4 ./project_2

