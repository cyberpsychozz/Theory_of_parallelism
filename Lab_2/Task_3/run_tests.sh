#!/bin/bash

# Пересборка проекта
make clean
make

# Массив с количеством потоков для тестирования
THREAD_COUNTS=(1 2 4 7 8 16 32 40)

# Размер матрицы (передается как аргумент программы)
MATRIX_SIZE=2000

echo "=== Запуск тестов производительности ==="

for threads in "${THREAD_COUNTS[@]}"
do
    echo "----------------------------------------"
    echo "Тест с потоками: $threads"
    echo "----------------------------------------"
    
    # Установка переменной окружения для OpenMP
    export OMP_NUM_THREADS=$threads
    
    # Запуск программы
    ./system_solution $MATRIX_SIZE
    
    echo ""
done