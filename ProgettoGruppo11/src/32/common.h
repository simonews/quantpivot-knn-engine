#ifndef QUANTPIVOT_COMMON
#define QUANTPIVOT_COMMON

#include <stdint.h>

#define	type	float
#define	align	16

#define	MATRIX		type*
#define	VECTOR		type*

typedef struct{
    // Variabili esistenti
    MATRIX DS;
    int* P;
    MATRIX index;
    MATRIX Q;
    int* id_nn;
    MATRIX dist_nn;
    
    // AGGIUNGI QUESTE 2 RIGHE:
    uint8_t* DS_quantized_plus;   // Dataset quantizzato [N × D]
    uint8_t* DS_quantized_minus;  // Dataset quantizzato [N × D]
    
    int h;
    int k;
    int x;
    int N;
    int D;
    int nq;
    int silent;
} params;

#endif
