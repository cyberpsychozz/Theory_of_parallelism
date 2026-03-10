#!/bin/bash

# Набор потоков для тестирования
THREADS=(1 2 4 7 8 16 20 40)

# Путь к исходному коду и исполняемому файлу
SRC="system_solution.cpp"
EXE="./system_solution"

echo "Начало тестирования производительности для $SRC"
# Заголовок CSV: потоки, время серийной версии, время параллельной (omp_one), ускорение, эффективность
echo "Threads,T_Serial,T_Parallel,Speedup,Efficiency" > results.csv

# Компилируем один раз перед циклом
g++ -O3 -fopenmp $SRC -o $EXE

for P in "${THREADS[@]}"
do
    echo "------------------------------------------"
    echo "Запуск с количеством потоков: $P"
    
    # Устанавливаем количество потоков через переменную окружения (так надежнее для OpenMP)
    export OMP_NUM_THREADS=$P
    
    # Запускаем и парсим вывод
    # Мы берем время именно из omp_one (как наиболее надежного варианта с одним параллельным блоком)
    OUTPUT=$($EXE)
    
    T_PARALLEL=$(echo "$OUTPUT" | grep "Время выполнения (omp_one):" | awk '{print $4}')
    
    # Если это первый запуск (P=1), сохраним это время как T_Serial для расчетов
    if [ "$P" -eq 1 ]; then
        T_SERIAL_BASE=$T_PARALLEL
    fi
    
    # Расчеты через bc (убедитесь, что он установлен)
    SPEEDUP=$(echo "scale=6; $T_SERIAL_BASE / $T_PARALLEL" | bc)
    EFFICIENCY=$(echo "scale=6; $SPEEDUP / $P" | bc)
    
    echo "T_Parallel: $T_PARALLEL, Speedup: $SPEEDUP, Efficiency: $EFFICIENCY"
    
    # Сохраняем в CSV
    echo "$P,$T_SERIAL_BASE,$T_PARALLEL,$SPEEDUP,$EFFICIENCY" >> results.csv
done

echo "------------------------------------------"
echo "Тестирование завершено. Результаты сохранены в Task_3/results.csv"
