#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace po = boost::program_options;

struct Options {
    int size = 128;
    int max_iterations = 1000000;
    int stage = 4;
    int check_interval = 10;
    double eps = 1e-6;
    std::string output = "result_gpu.dat";
    bool text_output = true;
    bool print_matrix = false;
};

struct SolveResult {
    int iterations = 0;
    double error = 0.0;
    double seconds = 0.0;
    const double* values = nullptr;
};

std::size_t idx(int row, int col, int n) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
           static_cast<std::size_t>(col);
}

double lerp(double left, double right, int pos, int last) {
    if (last == 0) {
        return left;
    }
    const double t = static_cast<double>(pos) / static_cast<double>(last);
    return left + (right - left) * t;
}

void initialize_grid(std::vector<double>& grid, int n) {
    std::fill(grid.begin(), grid.end(), 0.0);

    constexpr double top_left = 10.0;
    constexpr double top_right = 20.0;
    constexpr double bottom_right = 30.0;
    constexpr double bottom_left = 20.0;

    const int last = n - 1;
    for (int i = 0; i < n; ++i) {
        grid[idx(0, i, n)] = lerp(top_left, top_right, i, last);
        grid[idx(last, i, n)] = lerp(bottom_left, bottom_right, i, last);
        grid[idx(i, 0, n)] = lerp(top_left, bottom_left, i, last);
        grid[idx(i, last, n)] = lerp(top_right, bottom_right, i, last);
    }
}

void jacobi_step_no_error(const double* current, double* next, int n) {
#pragma acc parallel loop collapse(2) present(current[0:n * n], next[0:n * n])
    for (int row = 1; row < n - 1; ++row) {
        for (int col = 1; col < n - 1; ++col) {
            const int p = row * n + col;
            next[p] = 0.25 * (current[p - 1] + current[p + 1] + current[p - n] +
                              current[p + n]);
        }
    }
}

double jacobi_step_with_error(const double* current, double* next, int n) {
    double error = 0.0;

#pragma acc parallel loop collapse(2) present(current[0:n * n], next[0:n * n]) reduction(max:error)
    for (int row = 1; row < n - 1; ++row) {
        for (int col = 1; col < n - 1; ++col) {
            const int p = row * n + col;
            const double value =
                0.25 * (current[p - 1] + current[p + 1] + current[p - n] + current[p + n]);
            const double diff = std::fabs(value - current[p]);
            if (diff > error) {
                error = diff;
            }
            next[p] = value;
        }
    }

    return error;
}

SolveResult solve_stage1_multicore_on_gpu(std::vector<double>& first,
                                          std::vector<double>& second, int n,
                                          int max_iterations, double eps) {
    const int total = n * n;
    double* current = first.data();
    double* next = second.data();
    int iterations = 0;
    double error = 0.0;

    const auto started = std::chrono::steady_clock::now();

    for (iterations = 0; iterations < max_iterations; ++iterations) {
        error = 0.0;

#pragma acc kernels loop collapse(2) copyin(current[0:total]) copy(next[0:total]) reduction(max:error)
        for (int row = 1; row < n - 1; ++row) {
            for (int col = 1; col < n - 1; ++col) {
                const int p = row * n + col;
                const double value = 0.25 * (current[p - 1] + current[p + 1] +
                                             current[p - n] + current[p + n]);
                const double diff = std::fabs(value - current[p]);
                if (diff > error) {
                    error = diff;
                }
                next[p] = value;
            }
        }

        std::swap(current, next);

        if (error < eps) {
            ++iterations;
            break;
        }
    }

    const auto finished = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = finished - started;
    return SolveResult{iterations, error, elapsed.count(), current};
}

SolveResult solve_stage2_start_acc(std::vector<double>& first, std::vector<double>& second,
                                   int n, int max_iterations, double eps) {
    const int total = n * n;
    double* current = first.data();
    double* next = second.data();
    int iterations = 0;
    double error = 0.0;

    const auto started = std::chrono::steady_clock::now();

    for (iterations = 0; iterations < max_iterations; ++iterations) {
        error = 0.0;

#pragma acc parallel loop collapse(2) copyin(current[0:total]) copy(next[0:total]) reduction(max:error)
        for (int row = 1; row < n - 1; ++row) {
            for (int col = 1; col < n - 1; ++col) {
                const int p = row * n + col;
                const double value = 0.25 * (current[p - 1] + current[p + 1] +
                                             current[p - n] + current[p + n]);
                const double diff = std::fabs(value - current[p]);
                if (diff > error) {
                    error = diff;
                }
                next[p] = value;
            }
        }

        std::swap(current, next);

        if (error < eps) {
            ++iterations;
            break;
        }
    }

    const auto finished = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = finished - started;
    return SolveResult{iterations, error, elapsed.count(), current};
}

SolveResult solve_stage3_data_region(std::vector<double>& first, std::vector<double>& second,
                                     int n, int max_iterations, double eps) {
    const int total = n * n;
    double* current = first.data();
    double* next = second.data();
    int iterations = 0;
    double error = 0.0;

    const auto started = std::chrono::steady_clock::now();

#pragma acc data copyin(current[0:total], next[0:total])
    {
        for (iterations = 0; iterations < max_iterations; ++iterations) {
            error = jacobi_step_with_error(current, next, n);
            std::swap(current, next);

            if (error < eps) {
                ++iterations;
                break;
            }
        }

#pragma acc update self(current[0:total])
    }

    const auto finished = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = finished - started;
    return SolveResult{iterations, error, elapsed.count(), current};
}

SolveResult solve_stage4_less_sync(std::vector<double>& first, std::vector<double>& second,
                                   int n, int max_iterations, double eps, int check_interval) {
    const int total = n * n;
    double* current = first.data();
    double* next = second.data();
    int iterations = 0;
    double error = 0.0;
    check_interval = std::max(1, check_interval);

    const auto started = std::chrono::steady_clock::now();

#pragma acc data copyin(current[0:total], next[0:total])
    {
        while (iterations < max_iterations) {
            for (int step = 1; step < check_interval && iterations < max_iterations; ++step) {
                jacobi_step_no_error(current, next, n);
                std::swap(current, next);
                ++iterations;
            }

            if (iterations >= max_iterations) {
                break;
            }

            error = jacobi_step_with_error(current, next, n);
            std::swap(current, next);
            ++iterations;

            if (error < eps) {
                break;
            }
        }

#pragma acc update self(current[0:total])
    }

    const auto finished = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = finished - started;
    return SolveResult{iterations, error, elapsed.count(), current};
}

void save_matrix(const std::string& path, const double* matrix, int n, bool as_text) {
    std::ofstream out(path, as_text ? std::ios::out : std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot open output file: " + path);
    }

    if (as_text) {
        out << std::setprecision(12);
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                if (col != 0) {
                    out << ' ';
                }
                out << matrix[idx(row, col, n)];
            }
            out << '\n';
        }
        return;
    }

    out.write(reinterpret_cast<const char*>(matrix),
              static_cast<std::streamsize>(sizeof(double) * n * n));
}

void print_matrix(const double* matrix, int n) {
    std::cout << std::fixed << std::setprecision(6);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            std::cout << std::setw(11) << matrix[idx(row, col, n)];
        }
        std::cout << '\n';
    }
}

Options parse_options(int argc, char** argv) {
    Options options;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "show help message")
        ("size,n", po::value<int>(&options.size)->default_value(options.size),
         "grid size N for an N x N matrix")
        ("eps,e", po::value<double>(&options.eps)->default_value(options.eps),
         "target precision")
        ("iterations,i",
         po::value<int>(&options.max_iterations)->default_value(options.max_iterations),
         "maximum number of iterations")
        ("stage,s", po::value<int>(&options.stage)->default_value(options.stage),
         "GPU optimization stage: 1 multicore-style GPU, 2 start OpenACC, 3 data region, 4 less sync")
        ("check-interval",
         po::value<int>(&options.check_interval)->default_value(options.check_interval),
         "stage 4 convergence check interval")
        ("output,o", po::value<std::string>(&options.output)->default_value(options.output),
         "result matrix file")
        ("binary", po::bool_switch()->default_value(false),
         "save result matrix as raw binary doubles instead of text")
        ("print,p", po::bool_switch(&options.print_matrix),
         "print result matrix to terminal");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") != 0) {
        std::cout << desc << '\n';
        std::exit(0);
    }

    po::notify(vm);
    options.text_output = !vm["binary"].as<bool>();

    if (options.size < 2) {
        throw std::invalid_argument("size must be at least 2");
    }
    if (options.max_iterations < 1) {
        throw std::invalid_argument("iterations must be positive");
    }
    if (options.eps <= 0.0) {
        throw std::invalid_argument("eps must be positive");
    }
    if (options.stage < 1 || options.stage > 4) {
        throw std::invalid_argument("stage must be in range 1..4");
    }
    if (options.check_interval < 1) {
        throw std::invalid_argument("check-interval must be positive");
    }

    return options;
}

std::string stage_comment(int stage) {
    switch (stage) {
        case 1:
            return "Реализация для multicore, запущенная на GPU";
        case 2:
            return "Стартовый вариант OpenACC, явная параллельная область";
        case 3:
            return "Добавлена область данных, матрицы остаются в памяти GPU";
        case 4:
            return "Реже вычисляется ошибка сходимости, меньше синхронизаций";
        default:
            return "неизвестный этап";
    }
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::size_t total =
            static_cast<std::size_t>(options.size) * static_cast<std::size_t>(options.size);

        std::vector<double> first(total);
        std::vector<double> second(total);
        initialize_grid(first, options.size);
        initialize_grid(second, options.size);

        SolveResult result;
        if (options.stage == 1) {
            result = solve_stage1_multicore_on_gpu(first, second, options.size,
                                                   options.max_iterations, options.eps);
        } else if (options.stage == 2) {
            result = solve_stage2_start_acc(first, second, options.size,
                                            options.max_iterations, options.eps);
        } else if (options.stage == 3) {
            result = solve_stage3_data_region(first, second, options.size,
                                              options.max_iterations, options.eps);
        } else {
            result = solve_stage4_less_sync(first, second, options.size, options.max_iterations,
                                            options.eps, options.check_interval);
        }

        save_matrix(options.output, result.values, options.size, options.text_output);

        std::cout << "Stage: " << options.stage << '\n'
                  << "Comment: " << stage_comment(options.stage) << '\n'
                  << "Iterations: " << result.iterations << '\n'
                  << "Error: " << std::setprecision(12) << result.error << '\n'
                  << "Time: " << std::fixed << std::setprecision(6) << result.seconds
                  << " s\n"
                  << "Saved matrix: " << options.output << '\n';

        if (options.print_matrix || options.size == 10 || options.size == 13) {
            print_matrix(result.values, options.size);
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
