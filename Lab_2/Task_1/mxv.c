#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#ifndef NUM_THREADS
#define NUM_THREADS 40
#endif

#ifndef M_SIZE
#define M_SIZE 40000
#endif

#ifndef N_SIZE
#define N_SIZE 40000
#endif

double duration_serial, duration_parallel;

double fetch_time() {
    struct timespec t_spec;
    timespec_get(&t_spec, TIME_UTC);
    return ((double)t_spec.tv_sec + (double)t_spec.tv_nsec * 1.e-9);
}

void calc_mv_serial(double *mat, double *vec, double *res, size_t rows, size_t cols) {
    for (int r = 0; r < rows; ++r) {
        double temp_sum = 0.0;
        for (int c = 0; c < cols; ++c) {
            temp_sum += mat[r * cols + c] * vec[c];
        }
        res[r] = temp_sum;
    }
}

void calc_mv_parallel(double *mat, double *vec, double *res, size_t rows, size_t cols) {
#pragma omp parallel num_threads(NUM_THREADS)
    {
        int total_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int chunk_size = rows / total_threads;
        int start_row = thread_id * chunk_size;
        int end_row = (thread_id == total_threads - 1) ? (rows - 1) : (start_row + chunk_size - 1);
        
        for (int r = start_row; r <= end_row; ++r) {
            double temp_sum = 0.0;
            for (int c = 0; c < cols; ++c) {
                temp_sum += mat[r * cols + c] * vec[c];
            }
            res[r] = temp_sum;
        }
    }
}

void verify_serial(size_t cols, size_t rows) {
    double *matrix = (double*)malloc(sizeof(*matrix) * rows * cols);
    double *vector = (double*)malloc(sizeof(*vector) * cols);
    double *result = (double*)malloc(sizeof(*result) * rows);

    if (!matrix || !vector || !result) {
        free(matrix); free(vector); free(result);
        fprintf(stderr, "Fatal: Out of memory!\n");
        exit(EXIT_FAILURE);
    }

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            matrix[r * cols + c] = (double)(r + c);
        }
    }

    for (size_t c = 0; c < cols; ++c) {
        vector[c] = (double)c;
    }

    duration_serial = fetch_time();
    calc_mv_serial(matrix, vector, result, rows, cols);
    duration_serial = fetch_time() - duration_serial;

    printf("[Serial]   Time elapsed: %.6f seconds\n", duration_serial);
    
    free(matrix);
    free(vector);
    free(result);
}

void verify_parallel(size_t cols, size_t rows) {
    double *matrix = (double*)malloc(sizeof(*matrix) * rows * cols);
    double *vector = (double*)malloc(sizeof(*vector) * cols);
    double *result = (double*)malloc(sizeof(*result) * rows);

    if (!matrix || !vector || !result) {
        free(matrix); free(vector); free(result);
        fprintf(stderr, "Fatal: Out of memory!\n");
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int total_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int chunk_size = rows / total_threads;
        int start_row = thread_id * chunk_size;
        int end_row = (thread_id == total_threads - 1) ? (rows - 1) : (start_row + chunk_size - 1);
        
        for (int r = start_row; r <= end_row; ++r) {
            for (int c = 0; c < cols; ++c) {
                matrix[r * cols + c] = (double)(r + c);
            }
            result[r] = 0.0;
        }
    }

    for (size_t c = 0; c < cols; ++c) {
        vector[c] = (double)c;
    }

    duration_parallel = fetch_time();
    calc_mv_parallel(matrix, vector, result, rows, cols);
    duration_parallel = fetch_time() - duration_parallel;

    printf("[Parallel] Time elapsed: %.6f seconds\n", duration_parallel);
    
    free(matrix);
    free(vector);
    free(result);
}

int main(int argc, char *argv[]) {
    size_t rows_cnt = M_SIZE;
    size_t cols_cnt = N_SIZE;

    if (argc > 1) rows_cnt = (size_t)atol(argv[1]);
    if (argc > 2) cols_cnt = (size_t)atol(argv[2]);

    printf("--- Matrix-Vector Product ---\n");
    printf("Configuration: Matrix %zu x %zu, OpenMP threads = %d\n\n", rows_cnt, cols_cnt, NUM_THREADS);

    verify_serial(cols_cnt, rows_cnt);
    verify_parallel(cols_cnt, rows_cnt);
    
    printf("\nOverall speedup factor: %.6fx\n", duration_serial / duration_parallel);

    return 0;
}
