#!/bin/bash

# Набор потоков для тестирования
THREADS=(1 2 4 7 8 16 20 40)

# Путь к исходному коду и исполняемому файлу
SRC="integrate.c"
EXE="./integrate"

echo "Начало тестирования производительности для $SRC"
echo "Threads,Serial_Time,Parallel_Time,Speedup" > results.csv

for P in "${THREADS[@]}"
do
    echo "------------------------------------------"
    echo "Запуск с количеством потоков: $P"
    
    # Компилируем с передачей NUM_THREADS через макрос
    # -DNUM_THREADS=$P переопределяет значение в коде
    gcc -O3 -fopenmp -DNUM_THREADS=$P $SRC -o $EXE -lm
    
    # Запускаем и парсим вывод
    # Мы используем временный файл, чтобы вытащить нужные цифры
    OUTPUT=$($EXE)
    
    T_SERIAL=$(echo "$OUTPUT" | grep "Execution time (serial):" | awk '{print $4}')
    T_PARALLEL=$(echo "$OUTPUT" | grep "Execution time (parallel):" | awk '{print $4}')
    S_SPEEDUP=$(echo "$OUTPUT" | grep "Speedup:" | awk '{print $2}')
    
    # Вывод в консоль для наглядности
    echo "Serial: $T_SERIAL, Parallel: $T_PARALLEL, Speedup: $S_SPEEDUP"
    
    # Сохраняем в CSV
    echo "$P,$T_SERIAL,$T_PARALLEL,$S_SPEEDUP" >> results.csv
done

echo "------------------------------------------"
echo "Тестирование завершено. Результаты сохранены в results.csv"
