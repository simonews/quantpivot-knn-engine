CC = gcc

CFLAGS = -m64 -O3 -march=native -Wall -g -fopenmp

LIBS = -lm -fopenmp

all: main64

main64: main.c common.h quantpivot64.c
	$(CC) $(CFLAGS) main.c $(LIBS) -o main64

clean:
	rm -f *.o main64

.PHONY: all clean
