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
void richardson_serial(const std::vector<double>& A, const std::vector<double>& b, std::vector<double>& x, int n, double b_norm)
{
    const double tau = 0.01;
    int iter = 0;
    double residual_norm = 1.0;

    while (iter < MAX_ITER && (residual_norm / b_norm) > EPS){
        residual_norm = 0.0;
        for (int i = 0; i < n; i++){

            double Ax = 0.0;
            for (int j = 0; j < n; j++){
                Ax += A[i * n + j] * x[j];
            }

            double r = b[i] - Ax;
            x[i] += tau * r;
            residual_norm += r * r;
        }
        residual_norm = std::sqrt(residual_norm);
        iter++;
    }
    std::cout << "Метод простой итерации (serial) завершился за " << iter << " итераций, норма = " << std::scientific << (residual_norm / b_norm) << "\n";
}

void run_serial(const std::vector<double>& A, const std::vector<double>& b, int n, double b_norm)
{
    std::vector<double> x(n, 0.0);

    double t = cpuSecond();
    richardson_serial(A, b, x, n, b_norm);
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
void richardson_omp_separate(const std::vector<double>& A, const std::vector<double>& b, std::vector<double>& x, int n, double b_norm)
{
    const double tau = 0.01;
    int iter = 0;
    double residual_norm = 1.0;

    int num_threads = omp_get_max_threads();

    while (iter < MAX_ITER && (residual_norm / b_norm) > EPS){
        residual_norm = 0.0;

#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < n; i++){

            double Ax = 0.0;
            for (int j = 0; j < n; j++){
                Ax += A[i*n + j] * x[j];
            }
            
            double r = b[i] - Ax;
            x[i] += tau * r;
        }

#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < n; i++){
            
            double r = b[i];
            for (int j = 0; j < n; j++) r -= A[i*n + j] * x[j];
            double diff = r * r;

#pragma omp atomic
            residual_norm += diff;
        }

        residual_norm = std::sqrt(residual_norm);
        iter++;
    }
    std::cout << "Метод простой итерации (separate parallel for) завершился за " << iter << " итераций, норма = " << std::scientific << (residual_norm / b_norm) << "\n";
}

void run_omp_separate(const std::vector<double>& A, const std::vector<double>& b, int n, double b_norm)
{
    std::vector<double> x(n, 0.0);

    double t = cpuSecond();
    richardson_omp_separate(A, b, x, n, b_norm);
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
void richardson_omp_one(const std::vector<double>& A, const std::vector<double>& b, std::vector<double>& x, int n, double b_norm)
{
    const double tau = 0.01;
    int iter = 0;
    double residual_norm = 1.0;

    int num_threads = omp_get_max_threads();

    while (iter < MAX_ITER && (residual_norm / b_norm) > EPS){
        residual_norm = 0.0;

#pragma omp parallel num_threads(num_threads)
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
            residual_norm += norm_local;
        }
        residual_norm = std::sqrt(residual_norm);
        iter++;
    }
    std::cout << "Метод простой итерации (один parallel) завершился за " << iter << " итераций, норма = " << std::scientific << (residual_norm / b_norm) << "\n";
}

void run_omp_one(const std::vector<double>& A, const std::vector<double>& b, int n, double b_norm)
{
    std::vector<double> x(n, 0.0);

    double t = cpuSecond();
    richardson_omp_one(A, b, x, n, b_norm);
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
void richardson_omp_schedule(const std::vector<double>& A, const std::vector<double>& b, std::vector<double>& x, int n, const char* schedule_name, double b_norm)
{
    const double tau = 0.01;
    int iter = 0;
    double residual_norm = 1.0;

    std::cout << "schedule = " << schedule_name << std::endl;

    while (iter < MAX_ITER && (residual_norm / b_norm) > EPS) {
        residual_norm = 0.0;

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
            residual_norm += diff;
        }

        residual_norm = std::sqrt(residual_norm);
        iter++;
    }
    std::cout << "Завершился за " << iter << " итераций\n";
}

void run_schedule_test(const std::vector<double>& A, const std::vector<double>& b, int n, const char* schedule_name, double b_norm)
{
    std::vector<double> x(n, 0.0);


    double t = cpuSecond();
    richardson_omp_schedule(A, b, x, n, schedule_name, b_norm);
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

    int num_threads = NUM_THREADS;
    char* env_threads = std::getenv("OMP_NUM_THREADS");
    if (env_threads) num_threads = std::atoi(env_threads);

    std::vector<double> A(n * n);
    std::vector<double> b(n);

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[i * n + j] = (i == j) ? 2.0 : 1.0;
        }
        b[i] = n + 1;
    }

    double b_norm = 0.0;
    for (double val : b) b_norm += val * val;
    b_norm = std::sqrt(b_norm);

    std::cout << "Решение системы Ax=b методом простой итерации (Ричардсона)\n";
    std::cout << "Размерность N = " << n << ", MAX_ITER = " << MAX_ITER << ", EPS = " << EPS << ", Потоков = " << num_threads << "\n\n";

    run_omp_separate(A, b, n, b_norm);
    run_omp_one(A, b, n, b_norm);

    
    std::cout << "=== Исследование schedule для метода Ричардсона ===\n";
    std::cout << "N = " << n << " | Потоков = " << NUM_THREADS << "\n\n";

    // run_schedule_test(A, b, n, "static", b_norm);
    // run_schedule_test(A, b, n, "static,1", b_norm);
    // run_schedule_test(A, b, n, "static,32", b_norm);
    // run_schedule_test(A, b, n, "static,128", b_norm);
    // run_schedule_test(A, b, n, "dynamic", b_norm);
    // run_schedule_test(A, b, n, "dynamic,8", b_norm);
    // run_schedule_test(A, b, n, "dynamic,32", b_norm);
    // run_schedule_test(A, b, n, "guided", b_norm);
    // run_schedule_test(A, b, n, "guided,16", b_norm);
    // run_schedule_test(A, b, n, "auto", b_norm);

    return 0;
}
