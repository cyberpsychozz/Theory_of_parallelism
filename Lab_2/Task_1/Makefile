CC = gcc
CFLAGS = -Wall -Wextra -O3 -fopenmp

TARGET = mxv
SRC = mxv.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
