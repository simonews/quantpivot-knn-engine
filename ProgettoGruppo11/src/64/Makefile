CC = gcc
NASM = nasm

CFLAGS = -m64 -O3 -march=native -Wall -g -fopenmp
NASMFLAGS = -f elf64

LIBS = -lm -fopenmp

ASM_OBJ = quantpivot64_asm.o

all: main64

$(ASM_OBJ): quantpivot64.nasm
	$(NASM) $(NASMFLAGS) quantpivot64.nasm -o $(ASM_OBJ)

main64: main.c common.h quantpivot64.c $(ASM_OBJ)
	$(CC) $(CFLAGS) main.c $(ASM_OBJ) $(LIBS) -o main64

clean:
	rm -f *.o main64

.PHONY: all clean
