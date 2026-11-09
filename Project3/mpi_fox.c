
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <stdint.h>

/* Generate random numbers in (-1, 1] */
double random_number(void)
{
    return 2.0 * ((double)rand() + 1.0)
           / ((double)RAND_MAX + 1.0) - 1.0;
}

/* Pack matrix blocks into contiguous memory */
void pack_blocks(
    const double *M,
    double *packed,
    int N,
    int q,
    int block_size)
{
    int block_elements = block_size * block_size;

    for (int rank = 0; rank < q * q; rank++) {

        int block_row = rank / q;
        int block_col = rank % q;

        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {

                int global_i = block_row * block_size + i;
                int global_j = block_col * block_size + j;

                packed[rank * block_elements
                       + i * block_size + j] =
                    M[(size_t)global_i * N + global_j];
            }
        }
    }
}

/* Reconstruct the complete matrix from packed blocks */
void unpack_blocks(
    const double *packed,
    double *M,
    int N,
    int q,
    int block_size)
{
    int block_elements = block_size * block_size;

    for (int rank = 0; rank < q * q; rank++) {

        int block_row = rank / q;
        int block_col = rank % q;

        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {

                int global_i = block_row * block_size + i;
                int global_j = block_col * block_size + j;

                M[(size_t)global_i * N + global_j] =
                    packed[rank * block_elements
                           + i * block_size + j];
            }
        }
    }
}

/* Local block multiplication: C += A * B */
void local_multiply(
    const double *A,
    const double *B,
    double *C,
    int b)
{
    for (int i = 0; i < b; i++) {

        for (int k = 0; k < b; k++) {

            double aik = A[(size_t)i * b + k];

            for (int j = 0; j < b; j++) {

                C[(size_t)i * b + j] +=
                    aik * B[(size_t)k * b + j];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 2) {
        if (world_rank == 0)
            printf("Usage: %s N\n", argv[0]);

        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);

    if (N <= 0) {
        if (world_rank == 0)
            printf("N must be positive.\n");

        MPI_Finalize();
        return 1;
    }

    /* Fox requires a square number of processes */
    int q = (int)sqrt((double)world_size);

    if (q * q != world_size || N % q != 0) {

        if (world_rank == 0) {
            printf("Invalid configuration.\n");
            printf("Use a square number of processes: 4, 16, 64.\n");
            printf("N must be divisible by sqrt(P).\n");
        }

        MPI_Finalize();
        return 1;
    }

    int b = N / q;
    int block_elements = b * b;
    size_t total_elements = (size_t)N * N;

    /* Create Cartesian process grid */
    int dims[2] = {q, q};
    int periods[2] = {1, 0};
    int reorder = 0;

    MPI_Comm cart_comm;

    MPI_Cart_create(
        MPI_COMM_WORLD,
        2,
        dims,
        periods,
        reorder,
        &cart_comm
    );

    int rank, coords[2];

    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int row = coords[0];
    int col = coords[1];

    /* Create communicator for each process row */
    MPI_Comm row_comm;

    MPI_Comm_split(
        cart_comm,
        row,
        col,
        &row_comm
    );

    /*
       Each process has:
       A_initial: its original A block
       A_work:    broadcast A block
       B_block:   its current B block
       C_block:   its accumulated result
    */

    double *A_initial = malloc(
        (size_t)block_elements * sizeof(double));

    double *A_work = malloc(
        (size_t)block_elements * sizeof(double));

    double *B_block = malloc(
        (size_t)block_elements * sizeof(double));

    double *C_block = calloc(
        (size_t)block_elements, sizeof(double));

    if (A_initial == NULL || A_work == NULL ||
        B_block == NULL || C_block == NULL) {

        printf("Memory allocation failed on rank %d\n", rank);
        MPI_Abort(cart_comm, 1);
    }

    /* Full matrices and packed blocks are stored on root */
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    double *packed_A = NULL;
    double *packed_B = NULL;
    double *packed_C = NULL;

    if (rank == 0) {

        A = malloc(total_elements * sizeof(double));
        B = malloc(total_elements * sizeof(double));
        C = calloc(total_elements, sizeof(double));

        packed_A = malloc(total_elements * sizeof(double));
        packed_B = malloc(total_elements * sizeof(double));
        packed_C = malloc(total_elements * sizeof(double));

        if (A == NULL || B == NULL || C == NULL ||
            packed_A == NULL || packed_B == NULL ||
            packed_C == NULL) {

            printf("Root memory allocation failed.\n");
            MPI_Abort(cart_comm, 1);
        }

        /* Generate reproducible random matrices */
        srand(12345);

        for (size_t i = 0; i < total_elements; i++) {
            A[i] = random_number();
            B[i] = random_number();
        }

        /* Pack matrices into blocks */
        pack_blocks(A, packed_A, N, q, b);
        pack_blocks(B, packed_B, N, q, b);
    }

    MPI_Barrier(cart_comm);

    double start = MPI_Wtime();

    /* Distribute A and B blocks */
    MPI_Scatter(
        packed_A, block_elements, MPI_DOUBLE,
        A_initial, block_elements, MPI_DOUBLE,
        0, cart_comm
    );

    MPI_Scatter(
        packed_B, block_elements, MPI_DOUBLE,
        B_block, block_elements, MPI_DOUBLE,
        0, cart_comm
    );

    /*
       Find neighbors for cyclic upward B-block shifting.

       Each process sends B upward and receives
       the block from the process below it.
    */

    int source, destination;

    MPI_Cart_shift(
        cart_comm,
        0,
        -1,
        &source,
        &destination
    );

    /* Fox algorithm */
    for (int stage = 0; stage < q; stage++) {

        int k = (row + stage) % q;

        /*
           The process in column k owns the required
           original A block for this stage.

           Copy it into A_work before broadcasting,
           preserving A_initial for later stages.
        */

        if (col == k) {
            for (int x = 0; x < block_elements; x++)
                A_work[x] = A_initial[x];
        }

        /* Broadcast selected A block across this row */
        MPI_Bcast(
            A_work,
            block_elements,
            MPI_DOUBLE,
            k,
            row_comm
        );

        /* Multiply and accumulate local C block */
        local_multiply(
            A_work,
            B_block,
            C_block,
            b
        );

        /* Shift B upward for the next stage */
        if (stage < q - 1) {

            MPI_Sendrecv_replace(
                B_block,
                block_elements,
                MPI_DOUBLE,
                destination,
                0,
                source,
                0,
                cart_comm,
                MPI_STATUS_IGNORE
            );
        }
    }

    /* Gather local C blocks */
    MPI_Gather(
        C_block,
        block_elements,
        MPI_DOUBLE,
        packed_C,
        block_elements,
        MPI_DOUBLE,
        0,
        cart_comm
    );

    double local_time = MPI_Wtime() - start;
    double max_time = 0.0;

    /* Parallel execution time is the slowest process time */
    MPI_Reduce(
        &local_time,
        &max_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        cart_comm
    );

    if (rank == 0) {

        /* Reconstruct full C matrix */
        unpack_blocks(packed_C, C, N, q, b);

        /* Calculate checksum */
        double checksum = 0.0;

        for (size_t i = 0; i < total_elements; i++)
            checksum += C[i];

        printf("\nMPI Fox Matrix Multiplication\n");
        printf("Matrix size: %d x %d\n", N, N);
        printf("MPI processes: %d\n", world_size);
        printf("Process grid: %d x %d\n", q, q);
        printf("Block size: %d x %d\n", b, b);
        printf("Execution time: %.6f seconds\n", max_time);
        printf("Checksum: %.12e\n", checksum);

        free(A);
        free(B);
        free(C);
        free(packed_A);
        free(packed_B);
        free(packed_C);
    }

    free(A_initial);
    free(A_work);
    free(B_block);
    free(C_block);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();

    return 0;
}
