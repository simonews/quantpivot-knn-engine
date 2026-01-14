CC = gcc
NASM = nasm

# Flags di compilazione
CFLAGS = -m32 -O2 -march=native -msse3 -Wall -fopenmp
NASMFLAGS = -f elf32
LIBS = -lm -lgomp

# File sorgenti
ASM_SRC = quantpivot32.nasm
ASM_OBJ = quantpivot32_asm.o
MAIN_SRC = main.c
EXECUTABLE = quantpivot32

# Regola principale
all: $(EXECUTABLE)

# Compilazione assembly
$(ASM_OBJ): $(ASM_SRC)
	$(NASM) $(NASMFLAGS) $(ASM_SRC) -o $(ASM_OBJ)

# Linking finale
$(EXECUTABLE): $(ASM_OBJ) $(MAIN_SRC) common.h quantpivot32.c
	$(CC) $(CFLAGS) $(MAIN_SRC) $(ASM_OBJ) -o $(EXECUTABLE) $(LIBS)

# Pulizia
clean:
	rm -f $(ASM_OBJ) $(EXECUTABLE) out_*.ds2

# Test rapido (solo 10 query)
test: $(EXECUTABLE)
	./$(EXECUTABLE)

.PHONY: all clean test
