cmake_minimum_required(VERSION 3.10)
project(MatrixVectorMult)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Для работы с потоками в Linux необходимо добавить pthread
find_package(Threads REQUIRED)

add_executable(matrix_mult main.cpp)

# Линовка библиотеки потоков
target_link_libraries(matrix_mult PRIVATE Threads::Threads)
