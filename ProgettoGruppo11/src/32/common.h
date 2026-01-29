#ifndef QUANTPIVOT_COMMON
#define QUANTPIVOT_COMMON

#include <stdint.h>

// tipo
#define	type	float

// costante 
#define	align	16

// puntatori al tipo float 
#define	MATRIX		type*
#define	VECTOR		type*

typedef struct{
    
    MATRIX DS; // puntatore all'array del dataset
    int* P; // puntatore all'array con indici dei pivot
    MATRIX index; // puntatore alla matrice delle distanze pre-calcolate
    MATRIX Q; //puntatore all'array delle query 
    int* id_nn; // array di output per gli ID dei k vicini trovati
    MATRIX dist_nn; // array di output per le distanze dei k vicini trovati 
    
    // vettori quantizzati 
    uint8_t* DS_quantized_plus;   // Dataset quantizzato [N × D]
    uint8_t* DS_quantized_minus;  // Dataset quantizzato [N × D]
    
    int h; //numero di pivot da usare
    int k; // numero di vicini da trovare 
    int x; // fattore di quantizzazione (elementi massimi da considerare)
    int N; // numero totale di punti nel dataset
    int D; // dimensione di ogni punto (numero di features)
    int nq; // numero di query da processare 
    int silent; // flag booleano; 1->non stampa output dei dettagli (debug) 
} params;

#endif
