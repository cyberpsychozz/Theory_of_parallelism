CC = gcc
CFLAGS = -Wall -Wextra -O3 -fopenmp

TARGET = integrate
SRC = integrate.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	rm -f $(TARGET)
