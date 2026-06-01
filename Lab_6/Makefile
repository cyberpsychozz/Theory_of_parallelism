CXX = pgc++

CPU_SRC := therm.cpp
GPU_SRC := therm_gpu.cpp

COMMON_FLAGS ?= -O3 -std=c++17 -Minfo=all
HOST_ACC_FLAGS ?= -acc=host
MULTICORE_ACC_FLAGS ?= -acc=multicore
GPU_ACC_FLAGS ?= -acc=gpu
LDLIBS ?= -lboost_program_options

.PHONY: all clean run gpu host multicore cpu-gcc

all: therm_host therm_multicore therm_gpu

therm_host: $(CPU_SRC)
	$(CXX) $(COMMON_FLAGS) $(HOST_ACC_FLAGS) -o $@ $< $(LDLIBS)

therm_multicore: $(CPU_SRC)
	$(CXX) $(COMMON_FLAGS) $(MULTICORE_ACC_FLAGS) -o $@ $< $(LDLIBS)

therm_gpu: $(GPU_SRC)
	$(CXX) $(COMMON_FLAGS) $(GPU_ACC_FLAGS) -o $@ $< $(LDLIBS)

gpu: therm_gpu

host: therm_host

multicore: therm_multicore

run: therm_gpu
	./therm_gpu --size 128 --eps 1e-6 --iterations 1000000 --stage 4 --output result.dat

cpu-gcc:
	$(MAKE) clean
	$(MAKE) CXX=g++ COMMON_FLAGS="-O3 -std=c++17 -Wall -Wextra" \
		HOST_ACC_FLAGS= GPU_ACC_FLAGS= therm_host therm_gpu

clean:
	rm -f therm therm_host therm_multicore therm_gpu result.dat result_*.dat
	rm -rf __pycache__
