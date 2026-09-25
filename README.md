# MPI Project

## Project Description

This project implements a parallel program using the Message Passing Interface (MPI).

The program demonstrates communication between MPI processes and the execution of a parallel task across multiple processors.

## Source Code

The main source code is:

- `project1.c`

## Compilation

Compile the program in slurm script  using:

```bash
module load mpich/gcc/3.2.1

mpicc project.c -o project

mpirun -np 16 ./project
