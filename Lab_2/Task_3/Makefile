CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -fopenmp

TARGET = system_solution
SRC = system_solution.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
