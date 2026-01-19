CC = gcc

# Flag compilazione C
CFLAGS = -m64 -O3 -march=native -mavx -Wall -g -fopenmp

# Librerie
LIBS = -lm -fopenmp

all: main64omp

# Regola principale (solo C, no assembly)
main64omp: main.c common.h quantpivot64omp.c
	$(CC) $(CFLAGS) main.c $(LIBS) -o main64omp

clean:
	rm -f *.o main64omp out_*.ds2

test: main64omp
	./main64omp

.PHONY: all clean testCC = gcc

# Flag compilazione C
CFLAGS = -m64 -O3 -march=native -mavx -Wall -g -fopenmp

# Librerie
LIBS = -lm -fopenmp

all: main64omp

# Regola principale (solo C, no assembly)
main64omp: main.c common.h quantpivot64omp.c
	$(CC) $(CFLAGS) main.c $(LIBS) -o main64omp

clean:
	rm -f *.o main64omp out_*.ds2

test: main64omp
	./main64omp

.PHONY: all clean test
