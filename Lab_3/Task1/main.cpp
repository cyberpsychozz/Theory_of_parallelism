#include <iostream>
#include <vector>
#include <future>
#include <thread>
#include <chrono>
#include <algorithm>

// Функция для умножения части матрицы на вектор
void multiply_chunk(const std::vector<double>& matrix, const std::vector<double>& vec, 
                    std::vector<double>& result, int start_row, int end_row, int cols) {
    for (int i = start_row; i < end_row; ++i) {
        double sum = 0.0;
        for (int j = 0; j < cols; ++j) {
            sum += matrix[i * cols + j] * vec[j];
        }
        result[i] = sum;
    }
}

int main() {
    // Размеры матрицы (rows x cols)
    constexpr int rows = 5000;
    constexpr int cols = 5000;
    
    // Представление матрицы в виде одномерного массива (лучше работает с кэшем)
    std::vector<double> matrix(rows * cols);
    std::vector<double> vec(cols);
    std::vector<double> result(rows, 0.0);

    // Определяем доступное количество потоков аппаратной среды
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::cout << "Start computing on " << num_threads << " threads...\n";

    // --- 1. Параллельная инициализация массивов ---
    auto start_init = std::chrono::high_resolution_clock::now();
    std::vector<std::future<void>> init_futures;
    
    // Находим максимальную размерность, чтобы равномерно инициализировать и матрицу, и вектор
    int max_dim = std::max(rows, cols);
    int init_chunk_size = max_dim / num_threads; 
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_idx = t * init_chunk_size;
        int end_idx = (t == num_threads - 1) ? max_dim : start_idx + init_chunk_size;
        
        int s_row = std::min(start_idx, rows);
        int e_row = std::min(end_idx, rows);
        
        // Запуск асинхронных заданий на инициализацию
        init_futures.push_back(std::async(std::launch::async, 
            [s_row, e_row, cols, start_idx, end_idx, &matrix, &vec]() {
                // Инициализация матрицы
                for (int i = s_row; i < e_row; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        matrix[i * cols + j] = static_cast<double>(i + j);
                    }
                }
                // Инициализация вектора-множителя
                for (int i = start_idx; i < end_idx; ++i) {
                    if (i < cols) {
                        vec[i] = static_cast<double>(i) * 1.5;
                    }
                }
            }));
    }

    // Ожидание завершения всех заданий инициализации
    for (auto& f : init_futures) {
        f.get();
    }
    auto end_init = std::chrono::high_resolution_clock::now();
    
    
    // --- 2. Параллельное умножение матрицы на вектор ---
    auto start_mul = std::chrono::high_resolution_clock::now();
    std::vector<std::future<void>> mul_futures;
    int mul_chunk_size = rows / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_row = t * mul_chunk_size;
        int end_row = (t == num_threads - 1) ? rows : start_row + mul_chunk_size;

        mul_futures.push_back(std::async(std::launch::async, multiply_chunk, 
                                         std::cref(matrix), std::cref(vec), std::ref(result), 
                                         start_row, end_row, cols));
    }

    // Ожидание завершения всех заданий умножения
    for (auto& f : mul_futures) {
        f.get();
    }
    auto end_mul = std::chrono::high_resolution_clock::now();


    // --- Вывод времени работы и проверка корнер-кейсов ---
    std::chrono::duration<double> init_time = end_init - start_init;
    std::chrono::duration<double> mul_time = end_mul - start_mul;

    std::cout << "Initialization Time:   " << std::fixed << init_time.count() << " seconds\n";
    std::cout << "Multiplication Time:   " << std::fixed << mul_time.count() << " seconds\n";
    std::cout << "Sample result (0):     " << result[0] << "\n";
    std::cout << "Sample result (end):   " << result[rows - 1] << "\n";

    return 0;
}