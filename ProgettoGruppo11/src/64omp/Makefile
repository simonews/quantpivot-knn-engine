CC = gcc
CFLAGS = -m64 -O3 -march=native -mavx -Wall -g -fopenmp
LIBS = -lm -fopenmp

#Assembler NASM
ASM = nasm
ASMFLAGS = -f elf64

# Target principale
all: main64omp

# Compila solo main.c (che include quantpivot64omp.c) + assembly
main64omp: main.c common.h quantpivot64omp.c quantpivot64_asm.o
	$(CC) $(CFLAGS) -o $@ main.c quantpivot64_asm.o $(LIBS)

# Compila assembly AVX
quantpivot64_asm.o: quantpivot64.nasm
	$(ASM) $(ASMFLAGS) -o $@ $<

# Pulizia
clean:
	rm -f *.o main64omp out_*.ds2

# Test rapido
test: main64omp
	./main64omp

.PHONY: all clean test