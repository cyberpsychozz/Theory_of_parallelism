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

double t_serial, t_parallel;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

/*
 * matrix_vector_product: Compute matrix-vector product c[m] = a[m][n] * b[n]
 */

void matrix_vector_product(double *a, double *b, double *c, size_t m, size_t n)
{
    for (int i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

/*
    matrix_vector_product_omp: Compute matrix-vector product c[m] = a[m][n] * b[n]
*/
void matrix_vector_product_omp(double *a, double *b, double *c, size_t m, size_t n)
{
#pragma omp parallel num_threads(NUM_THREADS)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++)
        {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

void run_serial(size_t n, size_t m)
{
    double *a, *b, *c;
    a = (double*)malloc(sizeof(*a) * m * n);
    b = (double*)malloc(sizeof(*b) * n);
    c = (double*)malloc(sizeof(*c) * m);

    if (a == NULL || b == NULL || c == NULL)
    {
        free(a);
        free(b);
        free(c);
        printf("Error allocate memory!\n");
        exit(1);
    }

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (size_t j = 0; j < n; j++)
        b[j] = j;

    t_serial = cpuSecond();
    matrix_vector_product(a, b, c, m, n);
    t_serial = cpuSecond() - t_serial;

    printf("Elapsed time (serial): %.6f sec.\n", t_serial);
    free(a);
    free(b);
    free(c);
}

void run_parallel(size_t n, size_t m)
{


    double *a, *b, *c;

    a = (double*)malloc(sizeof(*a) * m * n);
    b = (double*)malloc(sizeof(*b) * n);
    c = (double*)malloc(sizeof(*c) * m);

    if (a == NULL || b == NULL || c == NULL)
    {
        free(a);
        free(b);
        free(c);
        printf("Error allocate memory!\n");
        exit(1);
    }

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < n; j++)
                a[i * n + j] = i + j;
            c[i] = 0.0;
        }
    }

    for (size_t j = 0; j < n; j++)  
        b[j] = j;
    
    t_parallel = cpuSecond();
    matrix_vector_product_omp(a, b, c, m, n);
    t_parallel = cpuSecond() - t_parallel;

    printf("Elapsed time (parallel): %.6f sec.\n", t_parallel);
    free(a);
    free(b);
    free(c);
}

int main(int argc, char *argv[])
{
    size_t M = M_SIZE;
    size_t N = N_SIZE;

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);

    printf("Запуск с M = %zu, N = %zu, потоков = %d\n", M, N, NUM_THREADS);

    run_serial(N, M);
    run_parallel(N, M);
    printf("Acceleration coeficient: %.6f\n", t_serial/t_parallel);

    return 0;
}
