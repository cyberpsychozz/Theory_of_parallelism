#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <omp.h>

#ifndef N_SIZE
#define N_SIZE 2000
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

#ifndef MAX_ITER
#define MAX_ITER 10000
#endif

#ifndef EPS
#define EPS 1e-5
#endif

double get_wall_time() {
    struct timespec time_spec;
    timespec_get(&time_spec, TIME_UTC);
    return static_cast<double>(time_spec.tv_sec) + static_cast<double>(time_spec.tv_nsec) * 1e-9;
}

void solve_serial(const std::vector<double>& matrix, const std::vector<double>& rhs, std::vector<double>& sol, int dim, double rhs_norm) {
    const double param_tau = 0.01;
    int step = 0;
    double current_residual = 1.0;

    while (step < MAX_ITER && (current_residual / rhs_norm) > EPS) {
        current_residual = 0.0;
        for (int i = 0; i < dim; ++i) {
            double mx = 0.0;
            for (int j = 0; j < dim; ++j) {
                mx += matrix[i * dim + j] * sol[j];
            }
            double diff = rhs[i] - mx;
            sol[i] += param_tau * diff;
            current_residual += diff * diff;
        }
        current_residual = std::sqrt(current_residual);
        step++;
    }
    std::cout << "[Serial] Completed in " << step << " iterations, rel_norm = " << std::scientific << (current_residual / rhs_norm) << "\n";
}

void test_serial(const std::vector<double>& matrix, const std::vector<double>& rhs, int dim, double rhs_norm) {
    std::vector<double> sol(dim, 0.0);

    double start_time = get_wall_time();
    solve_serial(matrix, rhs, sol, dim, rhs_norm);
    double end_time = get_wall_time() - start_time;

    double max_err = 0.0;
    for (int i = 0; i < dim; ++i) {
        double err = std::fabs(sol[i] - 1.0);
        if (err > max_err) max_err = err;
    }

    std::cout << "  Execution time: " << std::fixed << std::setprecision(6) << end_time << " s\n";
    std::cout << "  Max error: " << std::scientific << max_err << "\n\n";
}

void solve_omp_split(const std::vector<double>& matrix, const std::vector<double>& rhs, std::vector<double>& sol, int dim, double rhs_norm) {
    const double param_tau = 0.01;
    int step = 0;
    double current_residual = 1.0;
    int n_threads = omp_get_max_threads();

    while (step < MAX_ITER && (current_residual / rhs_norm) > EPS) {
        current_residual = 0.0;

#pragma omp parallel for num_threads(n_threads)
        for (int i = 0; i < dim; ++i) {
            double mx = 0.0;
            for (int j = 0; j < dim; ++j) {
                mx += matrix[i * dim + j] * sol[j];
            }
            sol[i] += param_tau * (rhs[i] - mx);
        }

#pragma omp parallel for num_threads(n_threads) reduction(+:current_residual)
        for (int i = 0; i < dim; ++i) {
            double diff = rhs[i];
            for (int j = 0; j < dim; ++j) {
                diff -= matrix[i * dim + j] * sol[j];
            }
            current_residual += diff * diff;
        }

        current_residual = std::sqrt(current_residual);
        step++;
    }
    std::cout << "[OMP Split For] Completed in " << step << " iterations, rel_norm = " << std::scientific << (current_residual / rhs_norm) << "\n";
}

void test_omp_split(const std::vector<double>& matrix, const std::vector<double>& rhs, int dim, double rhs_norm) {
    std::vector<double> sol(dim, 0.0);

    double start_time = get_wall_time();
    solve_omp_split(matrix, rhs, sol, dim, rhs_norm);
    double end_time = get_wall_time() - start_time;

    double max_err = 0.0;
    for (int i = 0; i < dim; ++i) {
        double err = std::fabs(sol[i] - 1.0);
        if (err > max_err) max_err = err;
    }

    std::cout << "  Execution time: " << std::fixed << std::setprecision(6) << end_time << " s\n";
    std::cout << "  Max error: " << std::scientific << max_err << "\n\n";
}

void solve_omp_single_region(const std::vector<double>& matrix, const std::vector<double>& rhs, std::vector<double>& sol, int dim, double rhs_norm) {
    const double param_tau = 0.01;
    int step = 0;
    double current_residual = 1.0;
    int n_threads = omp_get_max_threads();

    while (step < MAX_ITER && (current_residual / rhs_norm) > EPS) {
        current_residual = 0.0;

#pragma omp parallel num_threads(n_threads)
        {
            int t_count = omp_get_num_threads();
            int t_id = omp_get_thread_num();
            int chunk_size = dim / t_count;
            int start_idx = t_id * chunk_size;
            int end_idx = (t_id == t_count - 1) ? (dim - 1) : (start_idx + chunk_size - 1);

            double local_res = 0.0;
            for (int i = start_idx; i <= end_idx; ++i) {
                double mx = 0.0;
                for (int j = 0; j < dim; ++j) {
                    mx += matrix[i * dim + j] * sol[j];
                }
                double diff = rhs[i] - mx;
                sol[i] += param_tau * diff;
                local_res += diff * diff;
            }

#pragma omp atomic
            current_residual += local_res;
        }
        current_residual = std::sqrt(current_residual);
        step++;
    }
    std::cout << "[OMP Single Region] Completed in " << step << " iterations, rel_norm = " << std::scientific << (current_residual / rhs_norm) << "\n";
}

void test_omp_single_region(const std::vector<double>& matrix, const std::vector<double>& rhs, int dim, double rhs_norm) {
    std::vector<double> sol(dim, 0.0);

    double start_time = get_wall_time();
    solve_omp_single_region(matrix, rhs, sol, dim, rhs_norm);
    double end_time = get_wall_time() - start_time;

    double max_err = 0.0;
    for (int i = 0; i < dim; ++i) {
        double err = std::fabs(sol[i] - 1.0);
        if (err > max_err) max_err = err;
    }

    std::cout << "  Execution time: " << std::fixed << std::setprecision(6) << end_time << " s\n";
    std::cout << "  Max error: " << std::scientific << max_err << "\n\n";
}

void solve_omp_dyn_schedule(const std::vector<double>& matrix, const std::vector<double>& rhs, std::vector<double>& sol, int dim, const char* sched_type, double rhs_norm) {
    const double param_tau = 0.01;
    int step = 0;
    double current_residual = 1.0;

    std::cout << "Testing schedule scheme: " << sched_type << "\n";

    while (step < MAX_ITER && (current_residual / rhs_norm) > EPS) {
        current_residual = 0.0;

#pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
        for (int i = 0; i < dim; ++i) {
            double mx = 0.0;
            for (int j = 0; j < dim; ++j) {
                mx += matrix[i * dim + j] * sol[j];
            }
            sol[i] += param_tau * (rhs[i] - mx);
        }

#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:current_residual)
        for (int i = 0; i < dim; ++i) {
            double diff = rhs[i];
            for (int j = 0; j < dim; ++j) {
                diff -= matrix[i * dim + j] * sol[j];
            }
            current_residual += diff * diff;
        }

        current_residual = std::sqrt(current_residual);
        step++;
    }
    std::cout << "Finished in " << step << " iterations\n";
}

void test_omp_schedule(const std::vector<double>& matrix, const std::vector<double>& rhs, int dim, const char* sched_type, double rhs_norm) {
    std::vector<double> sol(dim, 0.0);

    double start_time = get_wall_time();
    solve_omp_dyn_schedule(matrix, rhs, sol, dim, sched_type, rhs_norm);
    double end_time = get_wall_time() - start_time;

    double max_err = 0.0;
    for (int i = 0; i < dim; ++i) {
        double err = std::fabs(sol[i] - 1.0);
        if (err > max_err) max_err = err;
    }

    std::cout << "  Execution time (" << sched_type << "): " << std::fixed << std::setprecision(6) << end_time << " s\n";
    std::cout << "  Max error (" << sched_type << "): " << std::scientific << max_err << "\n\n";
}

int main(int argc, char** argv) {
    int dim = N_SIZE;
    if (argc > 1) dim = std::atoi(argv[1]);

    int n_threads = NUM_THREADS;
    if (const char* env_t = std::getenv("OMP_NUM_THREADS")) {
        n_threads = std::atoi(env_t);
    }

    std::vector<double> matrix(dim * dim);
    std::vector<double> rhs(dim);

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            matrix[i * dim + j] = (i == j) ? 2.0 : 1.0;
        }
        rhs[i] = dim + 1.0;
    }


    double rhs_norm = 0.0;
    for (double v : rhs) rhs_norm += v * v;
    rhs_norm = std::sqrt(rhs_norm);

    std::cout << "--- Richardson Iteration Solver ---\n";
    std::cout << "Matrix dim (N) = " << dim << ", MAX_ITER = " << MAX_ITER << ", EPS = " << EPS << ", Threads = " << n_threads << "\n\n";

    test_omp_split(matrix, rhs, dim, rhs_norm);
    test_omp_single_region(matrix, rhs, dim, rhs_norm);

    std::cout << "=== Evaluation of OpenMP Schedules ===\n";
    std::cout << "N = " << dim << " | Thread Limit = " << NUM_THREADS << "\n\n";

    const char* schedules[] = {
        "static", "static,1", "static,32", "static,128",
        "dynamic", "dynamic,8", "dynamic,32",
        "guided", "guided,16", "auto"
    };

    for (const char* s : schedules) {
        test_omp_schedule(matrix, rhs, dim, s, rhs_norm);
    }

    return 0;
}
