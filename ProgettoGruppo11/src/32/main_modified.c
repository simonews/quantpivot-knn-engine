#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include <xmmintrin.h>

#include <omp.h>

#include "common.h"

#include "quantpivot32.c"
#include <stdint.h>


extern type euclidean_distance_c(const type* v, const type* w, int D);

/*
*
* 	load_data
* 	=========
*
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
	
	fp = fopen(filename, "rb");
	
	if (fp == NULL){
		printf("'%s': bad data file name!\n", filename);
		exit(0);
	}
	
	status = fread(&rows, sizeof(int), 1, fp);
	status = fread(&cols, sizeof(int), 1, fp);
	
	MATRIX data = _mm_malloc(rows*cols*sizeof(type), align);
	status = fread(data, sizeof(type), rows*cols, fp);
	fclose(fp);
	
	*n = rows;
	*k = cols;
	
	return data;
}

/*
* 	save_data
* 	=========
* 
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
		fwrite(&n, 4, 1, fp);
		fwrite(&k, 4, 1, fp);
		for (i = 0; i < n; i++) {
			fwrite(X, sizeof(type), k, fp);
			//printf("%i %i\n", ((int*)X)[0], ((int*)X)[1]);
			X += sizeof(type)*k;
		}
	}
	else{
		int x = 0;
		fwrite(&x, 4, 1, fp);
		fwrite(&x, 4, 1, fp);
	}
	fclose(fp);
}

// ============================================================================
// TEST ASSEMBLY - Confronta versione C vs Assembly
// ============================================================================
void test_assembly_correctness(params* input) {
	printf("\n========================================\n");
	printf("TEST CORRETTEZZA ASSEMBLY SSE\n");
	printf("========================================\n");
	
	// Test su alcune coppie di vettori dal dataset
	int num_tests = 10;
	if (num_tests > input->N) num_tests = input->N;
	if (num_tests > input->nq) num_tests = input->nq;
	
	int passed = 0;
	type max_diff = 0.0;
	
	printf("Testando %d coppie di vettori (D=%d)...\n\n", num_tests, input->D);
	
	for (int i = 0; i < num_tests; i++) {
		type* v = &input->DS[i * input->D];
		type* w = &input->Q[i * input->D];
		
		// Versione C
		type dist_c = euclidean_distance_c(v, w, input->D);
		
		// Versione Assembly
		type dist_asm = euclidean_distance(v, w, input->D);
		
		type diff = fabs(dist_c - dist_asm);
		if (diff > max_diff) max_diff = diff;
		
		int pass = (diff < 1e-5);
		if (pass) passed++;
		
		printf("Test %2d: C=%.6f, ASM=%.6f, Diff=%.9f [%s]\n",
		       i+1, dist_c, dist_asm, diff, pass ? "PASS" : "FAIL");
	}
	
	printf("\nRisultato: %d/%d test passati\n", passed, num_tests);
	printf("Differenza massima: %.9f\n", max_diff);
	
	if (passed == num_tests) {
		printf("✓ Implementazione assembly corretta!\n");
	} else {
		printf("✗ ATTENZIONE: alcuni test falliti!\n");
	}
	printf("========================================\n\n");
}

// ============================================================================
// BENCHMARK ASSEMBLY - Misura speedup
// ============================================================================
void benchmark_assembly(params* input) {
	printf("\n========================================\n");
	printf("BENCHMARK PERFORMANCE ASSEMBLY SSE\n");
	printf("========================================\n");
	
	const int ITERATIONS = 10000;
	printf("Iterazioni: %d\n", ITERATIONS);
	printf("Dimensione vettori: D=%d\n\n", input->D);
	
	// Prendi alcune coppie dal dataset
	int test_pairs = 100;
	if (test_pairs > input->N) test_pairs = input->N;
	if (test_pairs > input->nq) test_pairs = input->nq;
	
	// Benchmark versione C
	clock_t start_c = clock();
	for (int iter = 0; iter < ITERATIONS; iter++) {
		for (int i = 0; i < test_pairs; i++) {
			volatile type result = euclidean_distance_c(
				&input->DS[i * input->D],
				&input->Q[i * input->D],
				input->D
			);
		}
	}
	clock_t end_c = clock();
	double time_c = ((double)(end_c - start_c)) / CLOCKS_PER_SEC;
	
	// Benchmark versione Assembly
	clock_t start_asm = clock();
	for (int iter = 0; iter < ITERATIONS; iter++) {
		for (int i = 0; i < test_pairs; i++) {
			volatile type result = euclidean_distance(
				&input->DS[i * input->D],
				&input->Q[i * input->D],
				input->D
			);
		}
	}
	clock_t end_asm = clock();
	double time_asm = ((double)(end_asm - start_asm)) / CLOCKS_PER_SEC;
	
	double speedup = time_c / time_asm;
	
	printf("Tempo versione C:        %.4f sec\n", time_c);
	printf("Tempo versione Assembly: %.4f sec\n", time_asm);
	printf("Speedup:                 %.2fx\n", speedup);
	printf("========================================\n\n");
}

int main(int argc, char** argv) {

	// ================= Parametri di ingresso =================
	char* dsfilename = "../../../dataset_2000x256_32.ds2";
	char* queryfilename = "../../../query_2000x256_32.ds2";
	int h = 20;
	int k = 8;
	int x = 2;
	int silent = 0;
	int test_asm = 1;  // NUOVO: abilita test assembly

	// Parsing argomenti opzionali
	if (argc > 1) {
		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i], "--no-test") == 0) {
				test_asm = 0;
			} else if (strcmp(argv[i], "--silent") == 0) {
				silent = 1;
			}
		}
	}

	// =========================================================

	params* input = malloc(sizeof(params));

	input->DS = load_data(dsfilename, &input->N, &input->D);
	input->Q = load_data(queryfilename, &input->nq, &input->D);
	
	//input->nq = 10; //TEST: solo 10 query 
	//printf("ATTENZIONE: numero query ridotto per test\n");
	
	input->id_nn = _mm_malloc(input->nq*input->k*sizeof(int), align);
	input->dist_nn = _mm_malloc(input->nq*input->k*sizeof(type), align);
	input->h = h;
	input->k = k;
	input->x = x;
	input->silent = silent;

	printf("Dataset caricato: N=%d, D=%d\n", input->N, input->D);
	printf("Query caricate: nq=%d, D=%d\n", input->nq, input->D);

	// ============================================================================
	// NUOVO: Test correttezza e performance assembly (se abilitato)
	// ============================================================================
	if (test_asm && !silent) {
		test_assembly_correctness(input);
		benchmark_assembly(input);
	}

	clock_t t;
	float time;

	t = omp_get_wtime();
	// =========================================================
	fit(input);
	// =========================================================
	time = omp_get_wtime() - t;

	if(!input->silent)
		printf("FIT time = %.5f secs\n", time);
	else
		printf("%.3f\n", time);

	t = omp_get_wtime();
	// =========================================================
	predict(input);
	// =========================================================
	time = omp_get_wtime() - t;

	if(!input->silent)
		printf("PREDICT time = %.5f secs\n", time);
	else
		printf("%.3f\n", time);

	// Salva il risultato
	char* outname_id = "out_idnn.ds2";
	char* outname_k = "out_distnn.ds2";
	save_data(outname_id, input->id_nn, input->nq, input->k);
	save_data(outname_k, input->dist_nn, input->nq, input->k);

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

	_mm_free(input->DS);
	_mm_free(input->Q);
	_mm_free(input->P);
	_mm_free(input->index);
	_mm_free(input->id_nn);
	_mm_free(input->dist_nn);
	//aggiunta di due free
	_mm_free(input->DS_quantized_plus); 
	_mm_free(input->DS_quantized_minus);
	free(input);

	return 0;
}