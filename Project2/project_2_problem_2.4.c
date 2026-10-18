
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX_ITER 100
#define MAX_ROOTS 5000
#define N_SELECT 20

#define TOL 1e-8
#define DUP_TOL 1e-6
#define DET_TOL 1e-14
#define TOTAL_ATTEMPTS (10 * 15 * 11 * 11)

#define PI 3.14159265358979323846

/* --------------------------------------------------
   Newton's method for one nonlinear system
   -------------------------------------------------- */

int newton(double x0, double y0, int k, int m,
           double *root_x, double *root_y)
{
    double x = x0;
    double y = y0;

    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        double f = pow(x, 4) + x * pow(y, 2)
                   - k * PI / 10.0;

        double g = pow(y, 3) - pow(x, 2) * y
                   + x / 2.0 - m * PI / 10.0;

        if (!isfinite(f) || !isfinite(g))
            return 0;

        if (fabs(f) < TOL && fabs(g) < TOL)
        {
            *root_x = x;
            *root_y = y;
            return 1;
        }

        double j11 = 4.0 * pow(x, 3) + pow(y, 2);
        double j12 = 2.0 * x * y;
        double j21 = -2.0 * x * y + 0.5;
        double j22 = 3.0 * pow(y, 2) - pow(x, 2);

        double D = j11 * j22 - j12 * j21;

        if (!isfinite(D) || fabs(D) < DET_TOL)
            return 0;

        double dx = (-f * j22 + j12 * g) / D;
        double dy = (j21 * f - j11 * g) / D;

        x += dx;
        y += dy;

        if (!isfinite(x) || !isfinite(y))
            return 0;
    }

    return 0;
}

/* --------------------------------------------------
   Original function u(x,y)
   -------------------------------------------------- */

double original_function(double x, double y)
{
    double phi1 = pow(x, 4) + x * pow(y, 2);

    double phi2 = pow(y, 3) - pow(x, 2) * y
                  + x / 2.0;

    return 0.5 * (
        pow(sin(10.0 * phi1), 2) +
        pow(sin(10.0 * phi2), 2)
    );
}

/* --------------------------------------------------
   Validate a root against the original function
   -------------------------------------------------- */

int validate_root(double x, double y)
{
    if (!(x > -1.0 && x < 1.0 &&
          y > -1.0 && y < 1.0))
        return 0;

    double u = original_function(x, y);

    return isfinite(u) && u < 1e-14;
}

/* --------------------------------------------------
   Decode a global attempt ID
   -------------------------------------------------- */

void decode_id(int id, int *k, int *m, int *i, int *j)
{
    *j = id % 11;
    id /= 11;

    *i = id % 11;
    id /= 11;

    *m = (id % 15) - 7;
    id /= 15;

    *k = id - 3;
}

/* --------------------------------------------------
   Check whether a root is a duplicate
   -------------------------------------------------- */

int is_duplicate(double x, double y,
                 double roots_x[], double roots_y[],
                 int root_count)
{
    for (int i = 0; i < root_count; i++)
    {
        double dx = x - roots_x[i];
        double dy = y - roots_y[i];

        double distance = sqrt(dx * dx + dy * dy);

        if (distance < DUP_TOL)
            return 1;
    }

    return 0;
}

/* --------------------------------------------------
   Select 20 widely spread roots
   -------------------------------------------------- */

int select_roots(double roots_x[], double roots_y[],
                 int root_count, int selected_indices[])
{
    if (root_count < N_SELECT)
        return 0;

    int *selected = calloc(root_count, sizeof(int));

    if (selected == NULL)
        return 0;

    selected_indices[0] = 0;
    selected[0] = 1;

    for (int s = 1; s < N_SELECT; s++)
    {
        double largest_min_distance = -1.0;
        int best_index = -1;

        for (int i = 0; i < root_count; i++)
        {
            if (selected[i])
                continue;

            double min_distance = 1e100;

            for (int j = 0; j < s; j++)
            {
                int idx = selected_indices[j];

                double dx = roots_x[i] - roots_x[idx];
                double dy = roots_y[i] - roots_y[idx];

                double distance = sqrt(dx * dx + dy * dy);

                if (distance < min_distance)
                    min_distance = distance;
            }

            if (min_distance > largest_min_distance)
            {
                largest_min_distance = min_distance;
                best_index = i;
            }
        }

        if (best_index == -1)
        {
            free(selected);
            return 0;
        }

        selected_indices[s] = best_index;
        selected[best_index] = 1;
    }

    free(selected);
    return 1;
}

/* --------------------------------------------------
   Main MPI program
   -------------------------------------------------- */

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    /* Each rank stores its own locally unique roots. */
    double roots_x[MAX_ROOTS];
    double roots_y[MAX_ROOTS];

    int root_count = 0;
    int attempts = 0;
    int successful = 0;

    /* ----------------------------------------------
       1. Distribute the attempts among MPI ranks
       ---------------------------------------------- */

    for (int id = rank; id < TOTAL_ATTEMPTS; id += size)
    {
        int k, m, i, j;

        decode_id(id, &k, &m, &i, &j);

        double x0 = -0.9 + i * 0.18;
        double y0 = -0.9 + j * 0.18;

        double x, y;
        attempts++;

        if (!newton(x0, y0, k, m, &x, &y))
            continue;

        successful++;

        if (!validate_root(x, y))
            continue;

        if (is_duplicate(x, y, roots_x, roots_y, root_count))
            continue;

        if (root_count >= MAX_ROOTS)
        {
            fprintf(stderr,
                    "Rank %d: local root storage capacity reached.\n",
                    rank);

            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        roots_x[root_count] = x;
        roots_y[root_count] = y;
        root_count++;
    }



    /* ----------------------------------------------
       2. Reduce diagnostic counts to Rank 0
       ---------------------------------------------- */

    int total_attempts = 0;
    int total_successful = 0;

    MPI_Reduce(&attempts, &total_attempts, 1,
               MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&successful, &total_successful, 1,
               MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* ----------------------------------------------
       3. Gather each rank's root count
       ---------------------------------------------- */

    int *root_counts = NULL;
    int *displacements = NULL;

    if (rank == 0)
    {
        root_counts = malloc(size * sizeof(int));

        if (root_counts == NULL)
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Gather(&root_count, 1, MPI_INT,
               root_counts, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    /* ----------------------------------------------
       4. Calculate receive positions and allocate
          space for all candidate roots on Rank 0
       ---------------------------------------------- */

    int total_roots = 0;
    double *all_roots_x = NULL;
    double *all_roots_y = NULL;

    if (rank == 0)
    {
        displacements = malloc(size * sizeof(int));

        if (displacements == NULL)
            MPI_Abort(MPI_COMM_WORLD, 1);

        for (int r = 0; r < size; r++)
        {
            displacements[r] = total_roots;
            total_roots += root_counts[r];
        }

        /* Allocate at least one element, even if there
           are no candidates. */
        size_t capacity = (total_roots > 0)
                          ? (size_t)total_roots : 1;

        all_roots_x = malloc(capacity * sizeof(double));
        all_roots_y = malloc(capacity * sizeof(double));

        if (all_roots_x == NULL || all_roots_y == NULL)
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ----------------------------------------------
       5. Gather all x and y coordinates
       ---------------------------------------------- */

    MPI_Gatherv(roots_x, root_count, MPI_DOUBLE,
                all_roots_x, root_counts, displacements,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Gatherv(roots_y, root_count, MPI_DOUBLE,
                all_roots_y, root_counts, displacements,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ----------------------------------------------
       6. Rank 0 removes duplicates globally and
          selects the final 20 roots
       ---------------------------------------------- */

    if (rank == 0)
    {
        double *unique_x = NULL;
        double *unique_y = NULL;

        size_t capacity = (total_roots > 0)
                          ? (size_t)total_roots : 1;

        unique_x = malloc(capacity * sizeof(double));
        unique_y = malloc(capacity * sizeof(double));

        if (unique_x == NULL || unique_y == NULL)
            MPI_Abort(MPI_COMM_WORLD, 1);

        int unique_count = 0;

        for (int i = 0; i < total_roots; i++)
        {
            double x = all_roots_x[i];
            double y = all_roots_y[i];

            if (is_duplicate(x, y, unique_x, unique_y,
                             unique_count))
                continue;

            unique_x[unique_count] = x;
            unique_y[unique_count] = y;
            unique_count++;
        }

        printf("MPI processes: %d\n", size);
        printf("Total attempts: %d\n", total_attempts);
        printf("Converged attempts: %d\n", total_successful);
        printf("Total gathered local-unique roots: %d\n",
               total_roots);
        printf("Globally unique roots: %d\n", unique_count);

        int selected_indices[N_SELECT];

        if (select_roots(unique_x, unique_y,
                         unique_count, selected_indices))
        {
            printf("\n20 selected roots:\n");

            double minimum_distance = 1e100;

            for (int s = 0; s < N_SELECT; s++)
            {
                int idx = selected_indices[s];

                double x = unique_x[idx];
                double y = unique_y[idx];

                printf("%2d: x = %.10f, y = %.10f, u = %.3e\n",
                       s + 1, x, y,
                       original_function(x, y));

                for (int t = 0; t < s; t++)
                {
                    int previous_idx = selected_indices[t];

                    double dx = x - unique_x[previous_idx];
                    double dy = y - unique_y[previous_idx];

                    double distance = sqrt(dx * dx + dy * dy);

                    if (distance < minimum_distance)
                        minimum_distance = distance;
                }
            }

            printf("\nMinimum distance between selected roots: %.10f\n\n",
                   minimum_distance);
        }
        else
        {
            printf("\nCould not select 20 roots. "
                   "Only %d globally unique roots were found.\n",
                   unique_count);
        }

        free(unique_x);
        free(unique_y);
        free(all_roots_x);
        free(all_roots_y);
        free(root_counts);
        free(displacements);
    }

    double end_time = MPI_Wtime();
    double local_elapsed1 = end_time - start_time;


    /*Printing in order*/

    if (rank == 0){
        printf("Rank %d, execution time: %.8f seconds,local unique roots: %d\n\n",
        rank, local_elapsed1, root_count);
        MPI_Send(&rank, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
    }
    else{
        int number = 0;
        MPI_Recv(&number, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d, execution time: %.8f seconds,local unique roots: %d\n\n",
               rank, local_elapsed1, root_count);
        if (rank < size - 1) {
            MPI_Send(&rank, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
        }
    }



    MPI_Finalize();
    return 0;
}
