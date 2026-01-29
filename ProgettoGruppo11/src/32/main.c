#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <time.h>

#include <xmmintrin.h> // _mm_malloc e _mm_free

#include <omp.h> // funzioni di tempo

#include "common.h"

#include "quantpivot32.c"

/*
*	Legge da file una matrice di N righe
* 	e M colonne e la memorizza in un array lineare in row-major order
*
* 	Codifica del file:
* 	primi 4 byte: numero di righe (N) --> numero intero
* 	successivi 4 byte: numero di colonne (M) --> numero intero
* 	successivi N*M*4 byte: matrix data in row-major order --> numeri floating-point a precisione singola
*/
MATRIX load_data(char* filename, int *n, int *k) {
	FILE* fp;
	int rows, cols, status, i;
	
	fp = fopen(filename, "rb"); // lettura binaria
	
	if (fp == NULL){
		printf("'%s': bad data file name!\n", filename);
		exit(0);
	}
	
	// legge i primi 4 byte (righe)
	status = fread(&rows, sizeof(int), 1, fp);
	// legge i successivi 4 byte (colonne)
	status = fread(&cols, sizeof(int), 1, fp);
	
	/*
	*  Allocazione allineata
	*  Alloca (rows * cols * 4 Byte) allineando l'indirizzo a 16 byte 
	*  Ideale per blocchi a 128 bit
	*/
	MATRIX data = _mm_malloc(rows*cols*sizeof(type), align);

	// legge tutto il blocco di dati float insieme
	status = fread(data, sizeof(type), rows*cols, fp);
	fclose(fp);
	
	*n = rows;
	*k = cols;
	
	return data;
}

/*
*	Salva su file un array lineare in row-major order
*	come matrice di N righe e M colonne
* 
* 	Codifica del file:
* 	primi 4 byte: numero di righe (N) --> numero intero a 32 bit
* 	successivi 4 byte: numero di colonne (M) --> numero intero a 32 bit
* 	successivi N*M*4 byte: matrix data in row-major order --> numeri interi o floating-point a precisione singola
*/
void save_data(char* filename, void* X, int n, int k) {
	FILE* fp;
	int i;
	fp = fopen(filename, "wb");
	if(X != NULL){
		// scrive intestazione: N e K
		fwrite(&n, 4, 1, fp);
		fwrite(&k, 4, 1, fp);

		/*
		*  scrive i dati riga per riga
		*  type si riferisce a float (common.h) e funziona bene sia per gli ID
		*  che per le distanze 
		*/
		for (i = 0; i < n; i++) {
			fwrite(X, sizeof(type), k, fp);
			X += sizeof(type)*k; // riga successiva
		}
	}
	else{ // puntatore nullo
		int x = 0;
		fwrite(&x, 4, 1, fp);
		fwrite(&x, 4, 1, fp);
	}
	fclose(fp);
}

int main(int argc, char** argv) {
	// percorsi hardcoded ai file di datase te query 
	char* dsfilename = "../../../dataset_2000x256_32.ds2";
	char* queryfilename = "../../../query_2000x256_32.ds2";

	// parametri algoritmo 
	int h = 20; // numero pivot
	int k = 8; // numero di vicini da cercare 
	int x = 2; // fattore di quantizzazione 
	int silent = 0; // 0 --> stampa output a video 

	
	// alloca la struttura principale 
	params* input = malloc(sizeof(params));

	// carica dataset e query 
	input->DS = load_data(dsfilename, &input->N, &input->D);
	input->Q = load_data(queryfilename, &input->nq, &input->D);
	
	//input->nq = 10; //TEST: solo 10 query 
	//printf("ATTENZIONE: numero query ridotto per test\n");
	
	// allocazione memoria per i risultati (ID e distanze)
	input->id_nn = _mm_malloc(input->nq*input->k*sizeof(int), align);
	input->dist_nn = _mm_malloc(input->nq*input->k*sizeof(type), align);
	
	// copia dei parametri scalari nella struttura 
	input->h = h;
	input->k = k;
	input->x = x;
	input->silent = silent;



	printf("Dataset caricato: N=%d, D=%d\n", input->N, input->D);
	printf("Query caricate: nq=%d, D=%d\n", input->nq, input->D);

	
	// TEST ASSEMBLY vs C
	printf("\n TEST ASSEMBLY vs C \n");

	// Test 1: Vettori identici (distanza = 0)
	{
		type v[256], w[256];
		// inizializza vettori uguali 
		for (int i = 0; i < 256; i++) {
			v[i] = 1.5f;
			w[i] = 1.5f;
		}
		// chiamata alle rispettive versioni 
		type dist_c = euclidean_distance_c(v, w, 256);
		type dist_asm = euclidean_distance_asm(v, w, 256);
		// se maggiore di 0.000001 c'è un errore nell'Assembly
		printf("Test 1 (identici): C=%.9f, ASM=%.9f, diff=%.9e\n", 
		       dist_c, dist_asm, fabs(dist_c - dist_asm));
	}

	// Test rimanenti con vettori e dimensioni diverse per verificare i casi limite ("resto")


	// Test 2: Vettori con differenza costante
	{
		type v[256], w[256];
		for (int i = 0; i < 256; i++) {
			v[i] = 2.0f;
			w[i] = 1.0f;
		}
		type dist_c = euclidean_distance_c(v, w, 256);
		type dist_asm = euclidean_distance_asm(v, w, 256);
		printf("Test 2 (diff=1): C=%.9f, ASM=%.9f, diff=%.9e\n", 
		       dist_c, dist_asm, fabs(dist_c - dist_asm));
	}

	// Test 3: D non multiplo di 4 (testa residui)
	{
		type v[255], w[255];
		for (int i = 0; i < 255; i++) {
			v[i] = 2.0f;
			w[i] = 1.0f;
		}
		type dist_c = euclidean_distance_c(v, w, 255);
		type dist_asm = euclidean_distance_asm(v, w, 255);
		printf("Test 3 (D=255):  C=%.9f, ASM=%.9f, diff=%.9e\n\n", 
		       dist_c, dist_asm, fabs(dist_c - dist_asm));
	}

	// ESECUZIONE fit (Training/indicizzazione)

	clock_t t;
	float time;

	// misura il tmepo con OpenMP
	t = omp_get_wtime();

	// chiama fase fit (selezione pivot, calcolo indice)
	fit(input);
	
	time = omp_get_wtime() -t; // Calcolo Delta t

	if(!input->silent)
		printf("FIT time = %.5f secs\n", time);
	else
		printf("%.3f\n", time);


	// ESECUZIONE predict (querying)

	t = omp_get_wtime();

	// (ricerca k-nn per ogni query)
	predict(input);
	
	time = omp_get_wtime() - t;

	if(!input->silent)
		printf("PREDICT time = %.5f secs\n", time);
	else
		printf("%.3f\n", time);

	// Salva il risultato su disco locale
	char* outname_id = "out_idnn.ds2";
	char* outname_k = "out_distnn.ds2";
	save_data(outname_id, input->id_nn, input->nq, input->k);
	save_data(outname_k, input->dist_nn, input->nq, input->k);

	// se silent -->0 stampa a video i risultati per ogni query 
	if(!input->silent){
		for(int i=0; i<input->nq; i++){
			printf("ID NN Q%3i: ( ", i);
			for(int j=0; j<input->k; j++)
				printf("%i ", input->id_nn[i*input->k + j]);
			printf(")\n");
		}
		for(int i=0; i<input->nq; i++){
			printf("Dist NN Q%3i: ( ", i);
			for(int j=0; j<input->k; j++)
				printf("%f ", input->dist_nn[i*input->k + j]);
			printf(")\n");
		}
	}

	// libera tutta la memoria allineata allocata con _mm_malloc
	_mm_free(input->DS);
	_mm_free(input->Q);
	_mm_free(input->P);
	_mm_free(input->index);
	_mm_free(input->id_nn);
	_mm_free(input->dist_nn);
	// array quantizzati 
	_mm_free(input->DS_quantized_plus); 
	_mm_free(input->DS_quantized_minus);

	// libera la struttura principale 
	free(input);

	return 0;
}