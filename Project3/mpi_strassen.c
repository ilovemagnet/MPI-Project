
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>

/* Random number in (-1, 1] */
double random_number(void)
{
    return 2.0 * ((double)rand() + 1.0)
           / ((double)RAND_MAX + 1.0) - 1.0;
}

/* Ordinary matrix multiplication: C = A * B */
void serial_multiply(
    const double *A,
    const double *B,
    double *C,
    int n)
{
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {

            double aik = A[(size_t)i * n + k];

            for (int j = 0; j < n; j++) {
                C[(size_t)i * n + j] +=
                    aik * B[(size_t)k * n + j];
            }
        }
    }
}

/*
   Construct the seven Strassen operand pairs.

   Each output matrix has dimensions (n/2) x (n/2).
*/
void make_strassen_operands(
    const double *A,
    const double *B,
    int n,
    double *Aops[7],
    double *Bops[7])
{
    int h = n / 2;
    size_t count = (size_t)h * h;

    for (int t = 0; t < 7; t++) {
        Aops[t] = malloc(count * sizeof(double));
        Bops[t] = malloc(count * sizeof(double));

        if (Aops[t] == NULL || Bops[t] == NULL) {
            fprintf(stderr, "Strassen operand allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < h; j++) {

            double a11 = A[(size_t)i * n + j];
            double a12 = A[(size_t)i * n + j + h];
            double a21 = A[(size_t)(i + h) * n + j];
            double a22 = A[(size_t)(i + h) * n + j + h];

            double b11 = B[(size_t)i * n + j];
            double b12 = B[(size_t)i * n + j + h];
            double b21 = B[(size_t)(i + h) * n + j];
            double b22 = B[(size_t)(i + h) * n + j + h];

            size_t x = (size_t)i * h + j;

            /* M1 = (A11 + A22)(B11 + B22) */
            Aops[0][x] = a11 + a22;
            Bops[0][x] = b11 + b22;

            /* M2 = (A21 + A22)B11 */
            Aops[1][x] = a21 + a22;
            Bops[1][x] = b11;

            /* M3 = A11(B12 - B22) */
            Aops[2][x] = a11;
            Bops[2][x] = b12 - b22;

            /* M4 = A22(B21 - B11) */
            Aops[3][x] = a22;
            Bops[3][x] = b21 - b11;

            /* M5 = (A11 + A12)B22 */
            Aops[4][x] = a11 + a12;
            Bops[4][x] = b22;

            /* M6 = (A21 - A11)(B11 + B12) */
            Aops[5][x] = a21 - a11;
            Bops[5][x] = b11 + b12;

            /* M7 = (A12 - A22)(B21 + B22) */
            Aops[6][x] = a12 - a22;
            Bops[6][x] = b21 + b22;
        }
    }
}

/*
   Combine seven product matrices into one result matrix.

   Each M matrix is (n/2) x (n/2).
   C is n x n.
*/
void combine_strassen(
    const double *M[7],
    double *C,
    int n)
{
    int h = n / 2;

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < h; j++) {

            size_t x = (size_t)i * h + j;

            double m1 = M[0][x];
            double m2 = M[1][x];
            double m3 = M[2][x];
            double m4 = M[3][x];
            double m5 = M[4][x];
            double m6 = M[5][x];
            double m7 = M[6][x];

            /* C11 = M1 + M4 - M5 + M7 */
            C[(size_t)i * n + j] =
                m1 + m4 - m5 + m7;

            /* C12 = M3 + M5 */
            C[(size_t)i * n + j + h] =
                m3 + m5;

            /* C21 = M2 + M4 */
            C[(size_t)(i + h) * n + j] =
                m2 + m4;

            /* C22 = M1 - M2 + M3 + M6 */
            C[(size_t)(i + h) * n + j + h] =
                m1 - m2 + m3 + m6;
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0)
            printf("Usage: %s N levels\n", argv[0]);

        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    int levels = atoi(argv[2]);

    if (N <= 0 || (levels != 1 && levels != 2)) {
        if (rank == 0)
            printf("Use a positive N and levels = 1 or 2.\n");

        MPI_Finalize();
        return 1;
    }

    int divisor = (levels == 1) ? 2 : 4;

    if (N % divisor != 0) {
        if (rank == 0)
            printf("N must be divisible by %d.\n", divisor);

        MPI_Finalize();
        return 1;
    }

    int leaf_n = N / divisor;
    int task_count = (levels == 1) ? 7 : 49;

    size_t leaf_elements = (size_t)leaf_n * leaf_n;

    if (leaf_elements > (size_t)2147483647 / task_count) {
        if (rank == 0)
            printf("Problem is too large for MPI integer counts.\n");

        MPI_Finalize();
        return 1;
    }

    /* Distribute tasks as evenly as possible */
    int *counts = malloc((size_t)size * sizeof(int));
    int *displs = malloc((size_t)size * sizeof(int));

    if (counts == NULL || displs == NULL)
        MPI_Abort(MPI_COMM_WORLD, 1);

    int total_displacement = 0;

    for (int p = 0; p < size; p++) {

        int first = (task_count * p) / size;
        int last = (task_count * (p + 1)) / size;

        int local_tasks = last - first;

        counts[p] = (int)((size_t)local_tasks * leaf_elements);
        displs[p] = total_displacement;

        total_displacement += counts[p];
    }

    int first_task = (task_count * rank) / size;
    int last_task = (task_count * (rank + 1)) / size;
    int local_tasks = last_task - first_task;

    double *local_A = malloc(
        (size_t)(local_tasks > 0 ? local_tasks : 1)
        * leaf_elements * sizeof(double));

    double *local_B = malloc(
        (size_t)(local_tasks > 0 ? local_tasks : 1)
        * leaf_elements * sizeof(double));

    double *local_C = calloc(
        (size_t)(local_tasks > 0 ? local_tasks : 1)
        * leaf_elements, sizeof(double));

    if (local_A == NULL || local_B == NULL || local_C == NULL)
        MPI_Abort(MPI_COMM_WORLD, 1);

    /* Root owns the original matrices and task arrays */
    double *A = NULL;
    double *B = NULL;
    double *send_A = NULL;
    double *send_B = NULL;
    double *all_C = NULL;
    double *C = NULL;

    double *top_A[7] = {NULL};
    double *top_B[7] = {NULL};

    if (rank == 0) {

        size_t matrix_elements = (size_t)N * N;

        A = malloc(matrix_elements * sizeof(double));
        B = malloc(matrix_elements * sizeof(double));

        if (A == NULL || B == NULL)
            MPI_Abort(MPI_COMM_WORLD, 1);

        srand(12345);

        for (size_t i = 0; i < matrix_elements; i++) {
            A[i] = random_number();
            B[i] = random_number();
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    if (rank == 0) {

        size_t task_elements = (size_t)task_count * leaf_elements;

        send_A = malloc(task_elements * sizeof(double));
        send_B = malloc(task_elements * sizeof(double));

        if (send_A == NULL || send_B == NULL)
            MPI_Abort(MPI_COMM_WORLD, 1);

        /* First Strassen level */
        make_strassen_operands(A, B, N, top_A, top_B);

        if (levels == 1) {

            /* Seven tasks directly */
            for (int t = 0; t < 7; t++) {

                for (size_t x = 0; x < leaf_elements; x++) {
                    send_A[(size_t)t * leaf_elements + x] = top_A[t][x];
                    send_B[(size_t)t * leaf_elements + x] = top_B[t][x];
                }

                free(top_A[t]);
                free(top_B[t]);
            }
        }
        else {

            /*
               Second recursion level:
               Each first-level product produces seven
               smaller multiplication tasks.
            */
            for (int t = 0; t < 7; t++) {

                double *child_A[7] = {NULL};
                double *child_B[7] = {NULL};

                make_strassen_operands(
                    top_A[t],
                    top_B[t],
                    N / 2,
                    child_A,
                    child_B
                );

                for (int s = 0; s < 7; s++) {

                    int task = t * 7 + s;

                    for (size_t x = 0; x < leaf_elements; x++) {

                        send_A[(size_t)task * leaf_elements + x] =
                            child_A[s][x];

                        send_B[(size_t)task * leaf_elements + x] =
                            child_B[s][x];
                    }

                    free(child_A[s]);
                    free(child_B[s]);
                }

                free(top_A[t]);
                free(top_B[t]);
            }
        }

        /* Original matrices are no longer required */
        free(A);
        free(B);
        A = NULL;
        B = NULL;
    }

    /* Distribute A and B multiplication tasks */
    MPI_Scatterv(
        send_A, counts, displs, MPI_DOUBLE,
        local_A, counts[rank], MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    MPI_Scatterv(
        send_B, counts, displs, MPI_DOUBLE,
        local_B, counts[rank], MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    /* Every process calculates its assigned leaf products */
    for (int t = 0; t < local_tasks; t++) {

        serial_multiply(
            local_A + (size_t)t * leaf_elements,
            local_B + (size_t)t * leaf_elements,
            local_C + (size_t)t * leaf_elements,
            leaf_n
        );
    }

    if (rank == 0) {
        all_C = malloc(
            (size_t)task_count * leaf_elements * sizeof(double));

        if (all_C == NULL)
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Collect all leaf products */
    MPI_Gatherv(
        local_C, counts[rank], MPI_DOUBLE,
        all_C, counts, displs, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    if (rank == 0) {

        if (levels == 1) {

            /* Combine the seven products into C */
            const double *M[7];

            for (int t = 0; t < 7; t++)
                M[t] = all_C + (size_t)t * leaf_elements;

            C = calloc((size_t)N * N, sizeof(double));

            if (C == NULL)
                MPI_Abort(MPI_COMM_WORLD, 1);

            combine_strassen(M, C, N);
        }
        else {

            /*
               First combine the 49 leaf products into
               seven second-level results.
            */
            int top_n = N / 2;
            size_t top_elements = (size_t)top_n * top_n;

            double *top_results[7];

            for (int t = 0; t < 7; t++) {

                top_results[t] = calloc(
                    top_elements, sizeof(double));

                if (top_results[t] == NULL)
                    MPI_Abort(MPI_COMM_WORLD, 1);

                const double *M[7];

                for (int s = 0; s < 7; s++) {

                    int task = t * 7 + s;

                    M[s] = all_C + (size_t)task * leaf_elements;
                }

                combine_strassen(M, top_results[t], top_n);
            }

            /* Combine the seven first-level products */
            const double *M[7];

            for (int t = 0; t < 7; t++)
                M[t] = top_results[t];

            C = calloc((size_t)N * N, sizeof(double));

            if (C == NULL)
                MPI_Abort(MPI_COMM_WORLD, 1);

            combine_strassen(M, C, N);

            for (int t = 0; t < 7; t++)
                free(top_results[t]);
        }
    }

    double local_time = MPI_Wtime() - start;
    double max_time = 0.0;

    MPI_Reduce(
        &local_time,
        &max_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {

        double checksum = 0.0;
        size_t matrix_elements = (size_t)N * N;

        for (size_t i = 0; i < matrix_elements; i++)
            checksum += C[i];

        printf("\nMPI Strassen Matrix Multiplication\n");
        printf("Matrix size: %d x %d\n", N, N);
        printf("MPI processes: %d\n", size);
        printf("Recursion levels: %d\n", levels);
        printf("Leaf matrix size: %d x %d\n", leaf_n, leaf_n);
        printf("Total leaf tasks: %d\n", task_count);
        printf("Execution time: %.6f seconds\n", max_time);
        printf("Checksum: %.12e\n", checksum);

        free(send_A);
        free(send_B);
        free(all_C);
        free(C);
    }

    free(local_A);
    free(local_B);
    free(local_C);
    free(counts);
    free(displs);

    MPI_Finalize();

    return 0;
}
