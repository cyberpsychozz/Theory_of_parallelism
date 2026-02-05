1. Проверка задачи на работоспособность:
    1. Для запуска с типом float(по умолчанию):
    ```bash
    cd task_1
    cmake -B build
    cd build
    make
    ./main
    ```
    2. Для запуска с типом double:
    ```bash
    cd task_1
    cmake -B build -DUSE_DOUBLE=ON
    cd build
    make
    ./main
    ```
2. Вывод:
    1. Float: Sum: -0.0277862
    2. Double: Sum: 4.89582e-11
