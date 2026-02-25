CC = gcc
NASM = nasm

# Flags di compilazione per 64-bit
CFLAGS = -m64 -O2 -march=native -msse3 -Wall -fopenmp
NASMFLAGS = -f elf64
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

# Test rapido
test: $(EXECUTABLE)
	./$(EXECUTABLE)

# Info compilazione (per debug)
info:
	@echo "=== Configurazione Compilazione ==="
	@echo "Architettura: x86-64 (64-bit)"
	@echo "Assembly format: elf64"
	@echo "C compiler flags: $(CFLAGS)"
	@echo "NASM flags: $(NASMFLAGS)"
	@echo "==================================="

.PHONY: all clean test info