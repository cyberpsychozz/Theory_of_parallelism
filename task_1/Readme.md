Проверка задачи на работоспособность:
1. Склонируйте репозиторий: 
```bash
git clone https://github.com/cyberpsychozz/Theory_of_parallelism.git
```
2. Перейдите в папку с репозиторием в терминале:
```bash
cd task_1
```
3. Выберите конфигурацию:
Для запуска с типом float:
```bash
cmake -S . -B build
```
Для запуска с типом double:
```bash
cmake -S . -B build -DUSE_DOUBLE=ON
```
4. Запустите сборку:
```bash
cmake --build build
```
5. Запустите исполняемый файл:
```bash
 ./build/sinesum
```