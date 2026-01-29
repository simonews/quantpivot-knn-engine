#ifndef QUANTPIVOT_COMMON
#define QUANTPIVOT_COMMON

#include <stdint.h>

#define	type double
#define	align 32

#define	MATRIX		type*
#define	VECTOR		type*

typedef struct{
	// Variabili
	MATRIX DS; 					// dataset (qui array di double)
	int* P;						// vettore contenente gli indici dei pivot
	MATRIX index;				// indice (qui array di double)
	MATRIX Q;					// query (qui array di double)
	int* id_nn;					// ID dei vicini
	MATRIX dist_nn;				// distanze (qui array di double)

	
	// gli array restano uint8_t (quantizzazione binaria)
	uint8_t* DS_quantized_plus; 
	uint8_t* DS_quantized_minus; 


	int h;						// numero di pivot
	int k;						// numero di vicini
	int x;						// parametro x per la quantizzazione
	int N;						// numero di righe del dataset
	int D;						// numero di colonne/feature del dataset
	int nq;						// numero delle query
	int silent;					// modalità silenziosa
} params;

#endif
