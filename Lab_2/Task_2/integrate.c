#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#ifndef NUM_THREADS
#define NUM_THREADS 1
#endif

const double PI_CONST = 3.14159265358979323846;
const double limit_a = -4.0;
const double limit_b = 4.0;
const int total_steps = 40000000;

double get_runtime() {
    struct timespec t_spec;
    timespec_get(&t_spec, TIME_UTC);
    return ((double)t_spec.tv_sec + (double)t_spec.tv_nsec * 1.e-9);
}

double math_function(double val) {
    return exp(-val * val);
}

double compute_integral_serial(double (*func_ptr)(double), double start, double end, int steps) {
    double step_size = (end - start) / steps;
    double accumulated_sum = 0.0;

    for (int k = 0; k < steps; ++k) {
        accumulated_sum += func_ptr(start + step_size * (k + 0.5));
    }

    return accumulated_sum * step_size;
}

double compute_integral_omp(double (*func_ptr)(double), double start, double end, int steps) {
    double step_size = (end - start) / steps;
    double accumulated_sum = 0.0;

#pragma omp parallel num_threads(NUM_THREADS)
    {
        int num_t = omp_get_num_threads();
        int t_id = omp_get_thread_num();
        int chunk = steps / num_t;
        int lower_bound = t_id * chunk;
        int upper_bound = (t_id == num_t - 1) ? (steps - 1) : (lower_bound + chunk - 1);
        
        double local_sum = 0.0;

        for (int k = lower_bound; k <= upper_bound; ++k) {
            local_sum += func_ptr(start + step_size * (k + 0.5));
        }

        #pragma omp atomic
        accumulated_sum += local_sum;  
    }

    return accumulated_sum * step_size;
}

double eval_serial() {
    double start_time = get_runtime();
    double integral_res = compute_integral_serial(math_function, limit_a, limit_b, total_steps);
    double duration = get_runtime() - start_time;
    printf("[Serial] Result: %.12f; error: %.12f\n", integral_res, fabs(integral_res - sqrt(PI_CONST)));
    return duration;
}

double eval_parallel() {
    double start_time = get_runtime();
    double integral_res = compute_integral_omp(math_function, limit_a, limit_b, total_steps);
    double duration = get_runtime() - start_time;
    printf("[Parallel] Result: %.12f; error: %.12f\n", integral_res, fabs(integral_res - sqrt(PI_CONST)));
    return duration;
}

int main(int argc, char **argv) {
    printf("--- Integral Calculation for exp(-x^2) ---\n");
    printf("Range: [%.1f, %.1f], steps = %d\n", limit_a, limit_b, total_steps);
    printf("Configured threads: %d\n\n", NUM_THREADS);

    double duration_serial = eval_serial();
    double duration_parallel = eval_parallel();

    printf("Time (serial implementation):   %.6f sec\n", duration_serial);
    printf("Time (parallel implementation): %.6f sec\n", duration_parallel);
    printf("Calculated speedup: %.2fx\n", duration_serial / duration_parallel);
    
    return 0;
}
