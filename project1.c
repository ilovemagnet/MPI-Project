#include <mpi.h>
#include <stdio.h>


int main(int argc, char *argv[]){
    int rank, size;

    MPI_Init( &argc , &argv);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank==0){
        printf("Hello world from processor %d of %d processors. \n", rank, size);
        fflush(stdout);
        MPI_Send(&rank, 1, MPI_INT, rank+1, 0 , MPI_COMM_WORLD);
    }
    else{
        int number = 0;
        MPI_Recv(&number, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Hello world from processor %d of %d processors. \n", rank, size);
        fflush(stdout);
        if(rank<size-1){
            MPI_Send(&rank, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();

    return 0;
}
