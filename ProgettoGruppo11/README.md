# QuantPivot K-NN: Ricerca Ottimizzata dei K Vicini più Prossimi

## Descrizione

Implementazione ad alte prestazioni dell'algoritmo K-Nearest Neighbors che combina tecniche di pruning basate su pivot, quantizzazione sparsa, vettorizzazione SIMD (SSE/AVX) e parallelizzazione multi-thread (OpenMP). Raggiunge uno speedup fino a 14x rispetto alla versione C sequenziale.

## Caratteristiche Principali

- **Pruning con pivot**: Riduce le computazioni di distanza del 70-90% usando disuguaglianza triangolare
- **Quantizzazione sparsa**: Rappresentazione binaria dei vettori per calcolo approssimato rapido
- **Vettorizzazione SIMD**: Assembly ottimizzato con istruzioni SSE (32-bit) e AVX (64-bit)
- **Multi-threading**: Parallelizzazione OpenMP con scheduling dinamico e buffer privati per thread
- **Gestione residui**: Supporto corretto per dimensioni vettoriali arbitrarie (non multiple di 4)

## Performance

Benchmark su 2000 query, 2000 punti dataset, 256 dimensioni:

| Versione | Architettura | SIMD | Thread | Tempo FIT | Tempo PREDICT | Totale | Speedup |
|----------|--------------|------|--------|-----------|---------------|--------|---------|
| C Sequenziale | 64-bit | No | 1 | ~0.15s | ~0.25s | ~0.40s | 1.0x |
| OpenMP | 64-bit | No | 4 | 0.029s | 0.056s | 0.085s | 4.7x |
| AVX | 64-bit | AVX | 1 | 0.012s | 0.020s | 0.032s | 12.5x |
| **AVX + OpenMP** | **64-bit** | **AVX** | **4** | **0.004s** | **0.007s** | **0.011s** | **~14x** |

## Struttura Repository

```
.
├── src/
│   ├── 32bit/              # Implementazione 32-bit SSE (float)
│   ├── 64bit/              # Implementazione 64-bit AVX (double)
│   └── 64omp/              # Implementazione 64-bit AVX + OpenMP
├── dataset/                # Dataset di test (formato binario .ds2)
└── docs/                   # Documentazione dettagliata
```

## Requisiti

- **Compilatore**: GCC 7.0+ con supporto OpenMP
- **Assembler**: NASM 2.13+
- **CPU**: x86-64 con supporto AVX (Sandy Bridge o successivi)
- **Sistema Operativo**: Linux (testato su Ubuntu 20.04+)

## Compilazione ed Esecuzione

### Versione 64-bit AVX + OpenMP (Consigliata)

```bash
cd src/64omp
make clean && make

# Esegui con numero thread di default
./main64omp

# Esegui con numero thread specifico
OMP_NUM_THREADS=4 ./main64omp
```

### Altre Versioni

```bash
# 32-bit SSE
cd src/32bit && make && ./main32

# 64-bit AVX sequenziale
cd src/64bit && make && ./main64
```

## Algoritmo

### Fase 1: FIT (Costruzione Indice)

1. Selezione di h punti pivot dal dataset
2. Quantizzazione di tutti i vettori in rappresentazione binaria sparsa
3. Pre-calcolo delle distanze approssimate da ogni punto ai pivot

Complessità: O(N × h × D)

### Fase 2: PREDICT (Ricerca Query)

1. Quantizzazione del vettore query
2. Calcolo distanze query-pivot
3. Pruning: scarto candidati usando disuguaglianza triangolare
4. Raffinamento: calcolo distanza euclidea esatta sui K candidati
5. Ordinamento e restituzione K-NN

Complessità: O(nq × N × h) con pruning rate 70-90%

## Dettagli Implementazione SIMD

### Vettorizzazione Distanza Euclidea

Entrambe le versioni processano 4 elementi per iterazione:
- **32-bit SSE**: 4 float (registri XMM a 128-bit)
- **64-bit AVX**: 4 double (registri YMM a 256-bit)

### Gestione Elementi Residui

Per dimensioni non multiple di 4:
- Loop principale: processa D/4 iterazioni (divisione intera)
- Loop residui: gestisce i rimanenti D mod 4 elementi
- Accumulazione corretta con operazioni scalari separate

Esempio con D = 258:
- Loop vettoriale: 64 iterazioni × 4 = 256 elementi
- Loop residui: 2 iterazioni × 1 = 2 elementi
- Totale: 258 elementi (verificato con errore < 1e-15)

## Thread Safety (Versione OpenMP)

Garantita attraverso:
- **Buffer privati**: ogni thread alloca memoria di lavoro propria
- **Scritture disgiunte**: thread scrivono in regioni diverse dell'array output
- **Registri SIMD isolati**: YMM/XMM privati per ogni core
- **Scheduling dinamico**: gestisce carichi irregolari da pruning

## Formato Dati

File binari .ds2:
```
[4 byte] Numero righe (N)
[4 byte] Numero colonne (D)
[N×D×sizeof(type)] Dati matrice in row-major order
```

- Versione 32-bit: `type = float` (4 byte/elemento)
- Versione 64-bit: `type = double` (8 byte/elemento)

## Parametri Configurazione

Modificabili in `main.c`:
```c
int h = 20;    // Numero pivot (aumentare per dataset grandi)
int k = 8;     // Numero vicini da restituire
int x = 2;     // Parametro sparsità quantizzazione
```

Valori h consigliati:
- N = 10K: h = 30-50
- N = 100K: h = 50-75
- N = 500K: h = 75-100

## Scalabilità

| Dimensione Dataset | RAM | Tempo (4 thread) | Raccomandazione |
|-------------------|-----|------------------|-----------------|
| N < 10K | < 100 MB | < 1s | Versione sequenziale sufficiente |
| 10K-100K | 100 MB - 1 GB | 1-30s | Versione OpenMP ottimale |
| 100K-500K | 1-5 GB | 30s-5min | OpenMP con h aumentato |
| N > 500K | > 5 GB | > 5min | Considerare metodi approximate (LSH/HNSW) |

## Verifica Correttezza

L'implementazione include test automatici:

```bash
./main64omp

# Output atteso:
[TEST] Verifica euclidean_distance_asm...
      Distanza C:   6.408924176
      Distanza ASM: 6.408924176
      Differenza:   1.78e-15
      TEST PASSED
```

Tutte e tre le versioni sono verificate per:
- Correttezza con D mod 4 in {0, 1, 2, 3}
- Stabilità numerica (< 1e-6 per float, < 1e-15 per double)
- Thread safety sotto esecuzione concorrente

## Limitazioni Note

- Dataset deve stare in RAM (no supporto out-of-core)
- Solo distanza euclidea supportata
- Richiede supporto AVX della CPU (versioni 64-bit)
- Solo Linux (convenzioni ABI cdecl/System V)

## Possibili Estensioni

- Accelerazione GPU con CUDA/OpenCL
- Metriche di distanza aggiuntive (Manhattan, Cosine)
- Varianti approximate con parametri qualità/velocità
- Supporto per dati sparsi
- Implementazione distribuita per cluster

## Autori

- Andrea Attadia
- Vito Simone Goffredo
- Christian Iuele

Sviluppato nell'ambito del corso di Architetture degli Elaboratori, 2025.

## Licenza

Progetto fornito a scopo didattico. Nessuna garanzia è fornita.
