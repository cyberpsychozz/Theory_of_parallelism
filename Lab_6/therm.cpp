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
    double eps = 1e-6;
    std::string output = "result.dat";
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

SolveResult solve(std::vector<double>& first, std::vector<double>& second, int n,
                  int max_iterations, double eps) {
    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    double* current = first.data();
    double* next = second.data();
    int iterations = 0;
    double error = 0.0;

    const auto started = std::chrono::steady_clock::now();

#pragma acc data copyin(current[0:total], next[0:total])
    {
        for (iterations = 0; iterations < max_iterations; ++iterations) {
            error = 0.0;

#pragma acc parallel loop collapse(2) present(current[0:total], next[0:total]) reduction(max:error)
            for (int row = 1; row < n - 1; ++row) {
                for (int col = 1; col < n - 1; ++col) {
                    const std::size_t p = static_cast<std::size_t>(row) *
                                              static_cast<std::size_t>(n) +
                                          static_cast<std::size_t>(col);
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

    return options;
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

        const SolveResult result =
            solve(first, second, options.size, options.max_iterations, options.eps);

        save_matrix(options.output, result.values, options.size, options.text_output);

        std::cout << "Iterations: " << result.iterations << '\n'
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
