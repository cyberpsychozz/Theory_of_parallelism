#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <ctime>
#include <omp.h>

#ifndef N_SIZE
#define N_SIZE 10000
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 40
#endif

#ifndef MAX_ITER
#define MAX_ITER 10000
#endif

#ifndef EPS
#define EPS 1e-5
#endif

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

//###############################################РИЧАРДСОН SERIAL########################################################################
void richardson_serial(const std::vector<double>& A, const std::vector<double>& b, std::vector<double>& x, int n)
{
    const double tau = 0.01;
    int iter = 0;
    double norm = 1.0;

    while (iter < MAX_ITER && norm > EPS){
        norm = 0.0;
        for (int i = 0; i < n; i++){

            double Ax = 0.0;
            for (int j = 0; j < n; j++){
                Ax += A[i * n + j] * x[j];
            }

            double r = b[i] - Ax;
            x[i] += tau * r;
            norm += r * r;
        }
        norm = std::sqrt(norm);
        iter++;
    }
    std::cout << "Метод простой итерации (serial) завершился за " << iter << " итераций, норма = " << std::scientific << norm << "\n";
}

void run_serial(const std::vector<double>& A, const std::vector<double>& b, int n)
{
    std::vector<double> x(n, 0.0);

    double t = cpuSecond();
    richardson_serial(A, b, x, n);
    t = cpuSecond() - t;

    double max_err = 0.0;
    for (int i = 0; i < n; i++){
        if (std::fabs(x[i] - 1.0) > max_err) max_err = std::fabs(x[i] - 1.0);
    }

    std::cout << "Время выполнения (serial): " << std::fixed << std::setprecision(6) << t << " сек.\n";
    std::cout << "Максимальная ошибка (serial): " << std::scientific << max_err << "\n\n";
}
//###########################################################################################################################################




//###############################################РИЧАРДСОН SEPARATE########################################################################
void richardson_omp_separate(const std::vector<double>& A, const std::vector<double>& b, std::vector<double>& x, int n)
{
    const double tau = 0.01;
    int iter = 0;
    double norm = 1.0;

    while (iter < MAX_ITER && norm > EPS){
        norm = 0.0;

#pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < n; i++){

            double Ax = 0.0;
            for (int j = 0; j < n; j++){
                Ax += A[i*n + j] * x[j];
            }
            
            double r = b[i] - Ax;
            x[i] += tau * r;
        }

#pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < n; i++){
            
            double r = b[i];
            for (int j = 0; j < n; j++) r -= A[i*n + j] * x[j];
            double diff = r * r;

#pragma omp atomic
            norm += diff;
        }

        norm = std::sqrt(norm);
        iter++;
    }
    std::cout << "Метод простой итерации (separate parallel for) завершился за " << iter << " итераций, норма = " << std::scientific << norm << "\n";
}

void run_omp_separate(const std::vector<double>& A, const std::vector<double>& b, int n)
{
    std::vector<double> x(n, 0.0);

    double t = cpuSecond();
    richardson_omp_separate(A, b, x, n);
    t = cpuSecond() - t;

    double max_err = 0.0;
    for (int i = 0; i < n; i++){
        if (std::fabs(x[i] - 1.0) > max_err) max_err = std::fabs(x[i] - 1.0);
    }

    std::cout << "Время выполнения (omp_separate): " << std::fixed << std::setprecision(6) << t << " сек.\n";
    std::cout << "Максимальная ошибка (omp_separate): " << std::scientific << max_err << "\n\n";
}
//###########################################################################################################################################



//###############################################РИЧАРДСОН ONE########################################################################
void richardson_omp_one(const std::vector<double>& A, const std::vector<double>& b, std::vector<double>& x, int n)
{
    const double tau = 0.01;
    int iter = 0;
    double norm = 1.0;

    while (iter < MAX_ITER && norm > EPS){
        norm = 0.0;

#pragma omp parallel num_threads(NUM_THREADS)
       {
            int nthreads = omp_get_num_threads();
            int threadid = omp_get_thread_num();
            int items_per_thread = n / nthreads;
            int lb = threadid * items_per_thread;
            int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

            double norm_local = 0.0;
            for (int i = lb; i <= ub; i++){

                double Ax = 0.0;
                for (int j = 0; j < n; j++){
                    Ax += A[i*n + j] * x[j];
                }

                double r = b[i] - Ax;
                x[i] += tau * r;
                norm_local += r * r;
            }

            #pragma omp atomic
            norm += norm_local;
        }
        norm = std::sqrt(norm);
        iter++;
    }
    std::cout << "Метод простой итерации (один parallel) завершился за " << iter << " итераций, норма = " << std::scientific << norm << "\n";
}

void run_omp_one(const std::vector<double>& A, const std::vector<double>& b, int n)
{
    std::vector<double> x(n, 0.0);

    double t = cpuSecond();
    richardson_omp_one(A, b, x, n);
    t = cpuSecond() - t;

    double max_err = 0.0;
    for (int i = 0; i < n; i++){
        if (std::fabs(x[i] - 1.0) > max_err) max_err = std::fabs(x[i] - 1.0);
    }

    std::cout << "Время выполнения (omp_one): " << std::fixed << std::setprecision(6) << t << " сек.\n";
    std::cout << "Максимальная ошибка (omp_one): " << std::scientific << max_err << "\n\n";
}
//###########################################################################################################################################




//###############################################РИЧАРДСОН С SCHEDULE########################################################################
void richardson_omp_schedule(const std::vector<double>& A, const std::vector<double>& b, std::vector<double>& x, int n, const char* schedule_name)
{
    const double tau = 0.01;
    int iter = 0;
    double norm = 1.0;

    std::cout << "schedule = " << schedule_name << std::endl;

    while (iter < MAX_ITER && norm > EPS) {
        norm = 0.0;

#pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
        for (int i = 0; i < n; i++) {
            double Ax = 0.0;
            for (int j = 0; j < n; j++) {
                Ax += A[i * n + j] * x[j];
            }
            double r = b[i] - Ax;
            x[i] += tau * r;
        }

#pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < n; i++) {
            double r = b[i];
            for (int j = 0; j < n; j++) {
                r -= A[i * n + j] * x[j];
            }
            double diff = r * r;
#pragma omp atomic
            norm += diff;
        }

        norm = std::sqrt(norm);
        iter++;
    }
    std::cout << "Завершился за " << iter << " итераций\n";
}

void run_schedule_test(const std::vector<double>& A, const std::vector<double>& b, int n, const char* schedule_name)
{
    std::vector<double> x(n, 0.0);

    double t = cpuSecond();
    richardson_omp_schedule(A, b, x, n, schedule_name);
    t = cpuSecond() - t;

    double max_err = 0.0;
    for (int i = 0; i < n; i++){
        if (std::fabs(x[i] - 1.0) > max_err) max_err = std::fabs(x[i] - 1.0);
    }

    std::cout << "Время выполнения (omp_schedule): " << std::fixed << std::setprecision(6) << t << " сек.\n";
    std::cout << "Максимальная ошибка (omp_schedule): " << std::scientific << max_err << "\n\n";
}
//##########################################################################################################################################



int main(int argc, char** argv)
{
    int n = N_SIZE;
    if (argc > 1) n = std::atoi(argv[1]);

    std::vector<double> A(n * n);
    std::vector<double> b(n);

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[i * n + j] = (i == j) ? 2.0 : 1.0;
        }
        b[i] = n + 1;
    }

    std::cout << "Решение системы Ax=b методом простой итерации (Ричардсона)\n";
    std::cout << "Размерность N = " << n << ", MAX_ITER = " << MAX_ITER << ", EPS = " << EPS << "N_THREADS = " << NUM_THREADS << "\n\n";

    //run_serial(A, b, n);
    run_omp_separate(A, b, n);
    run_omp_one(A, b, n);

    
    std::cout << "=== Исследование schedule для метода Ричардсона ===\n";
    std::cout << "N = " << n << " | Потоков = " << NUM_THREADS << "\n\n";

    run_schedule_test(A, b, n, "static");
    run_schedule_test(A, b, n, "static,1");
    run_schedule_test(A, b, n, "static,32");
    run_schedule_test(A, b, n, "static,128");
    run_schedule_test(A, b, n, "dynamic");
    run_schedule_test(A, b, n, "dynamic,8");
    run_schedule_test(A, b, n, "dynamic,32");
    run_schedule_test(A, b, n, "guided");
    run_schedule_test(A, b, n, "guided,16");
    run_schedule_test(A, b, n, "auto");

    return 0;
}