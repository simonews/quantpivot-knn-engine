CC = gcc
NASM = nasm
CFLAGS = -m32 -O2 -march=native -msse -msse2 -Wall -fopenmp
NASMFLAGS = -f elf32
LIBS = -lm -lgomp

ASM_SRC = quantpivot32.nasm
ASM_OBJ = quantpivot32_asm.o
MAIN_SRC = main_modified.c
EXECUTABLE = quantpivot32

all: $(EXECUTABLE)

$(ASM_OBJ): $(ASM_SRC)
	$(NASM) $(NASMFLAGS) $(ASM_SRC) -o $(ASM_OBJ)

$(EXECUTABLE): $(ASM_OBJ) $(MAIN_SRC)
	$(CC) $(CFLAGS) $(MAIN_SRC) $(ASM_OBJ) -o $(EXECUTABLE) $(LIBS)

clean:
	rm -f $(ASM_OBJ) $(EXECUTABLE) out_*.ds2

.PHONY: all clean
