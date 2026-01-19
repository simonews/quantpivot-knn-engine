#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <xmmintrin.h>
#include <omp.h>

#include "common.h"
#include "quantpivot64omp.c"

/*
 * load_data
 * =========
 * Legge da file una matrice di N righe e M colonne
 * Memorizzazione in row-major order
 *
 * Codifica file:
 * - primi 4 byte: numero righe (N) → int
 * - successivi 4 byte: numero colonne (M) → int
 * - successivi N*M*sizeof(type) byte: dati matrice
 */
MATRIX load_data(char* filename, int *n, int *k) {
    FILE* fp;
    int rows, cols, status;
    
    fp = fopen(filename, "rb");
    
    if (fp == NULL){
        printf("'%s': bad data file name!\n", filename);
        exit(0);
    }
    
    status = fread(&rows, sizeof(int), 1, fp);
    status = fread(&cols, sizeof(int), 1, fp);
    
    printf("[DEBUG] File: %s, rows=%d, cols=%d, sizeof(type)=%zu\n",
           filename, rows, cols, sizeof(type));
    
    // Alloca e leggi come type (double)
    MATRIX data = _mm_malloc(rows * cols * sizeof(type), align);
    status = fread(data, sizeof(type), rows * cols, fp);
    fclose(fp);
    
    *n = rows;
    *k = cols;
    
    return data;
}

/*
 * save_data
 * =========
 * Salva array lineare come matrice N x M
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

/*
 * save_int_data
 * =============
 * Versione per array di interi (4 byte)
 */
void save_int_data(char* filename, int* X, int n, int k) {
    FILE* fp;
    int i;
    fp = fopen(filename, "wb");
    if(X != NULL){
        fwrite(&n, 4, 1, fp);
        fwrite(&k, 4, 1, fp);
        for (i = 0; i < n; i++) {
            fwrite(X, sizeof(int), k, fp);
            X += k; 
        }
    }
    else{
        int x = 0;
        fwrite(&x, 4, 1, fp);
        fwrite(&x, 4, 1, fp);
    }
    fclose(fp);
}

int main(int argc, char** argv) {

    // ================= Parametri di ingresso =================
    char* dsfilename = "../../../dataset_2000x256_64.ds2";
    char* queryfilename = "../../../query_2000x256_64.ds2";
    int h = 20;
    int k = 8;
    int x = 2;
    int silent = 0;
    // =========================================================

    params* input = malloc(sizeof(params));

    // Inizializza parametri
    input->h = h;
    input->k = k;
    input->x = x;
    input->silent = silent;

    input->DS = load_data(dsfilename, &input->N, &input->D);
    input->Q = load_data(queryfilename, &input->nq, &input->D);

    input->id_nn = _mm_malloc(input->nq*input->k*sizeof(int), align);
    input->dist_nn = _mm_malloc(input->nq*input->k*sizeof(type), align);

    input->P = NULL;
    input->index = NULL;
    input->DS_quantized_plus = NULL;
    input->DS_quantized_minus = NULL;

    printf("Dataset caricato: N=%d, D=%d\n", input->N, input->D);
    printf("Query caricate: nq=%d, D=%d\n", input->nq, input->D);
    printf("Thread OpenMP disponibili: %d\n", omp_get_max_threads());

    double t;
    
    // =========================================================
    // FIT
    // =========================================================
    t = omp_get_wtime();
    fit(input);
    t = omp_get_wtime() - t;

    if(!input->silent)
        printf("FIT time = %.5f secs\n", t);
    else
        printf("%.3f\n", t);

    // =========================================================
    // PREDICT
    // =========================================================
    t = omp_get_wtime();
    predict(input);
    t = omp_get_wtime() - t;

    if(!input->silent)
        printf("PREDICT time = %.5f secs\n", t);
    else
        printf("%.3f\n", t);

    // Salva risultati
    char* outname_id = "out_idnn.ds2";
    char* outname_k = "out_distnn.ds2";
    save_int_data(outname_id, input->id_nn, input->nq, input->k);
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

    // Cleanup
    _mm_free(input->DS);
    _mm_free(input->Q);
    if (input->P) _mm_free(input->P);
    if (input->index) _mm_free(input->index);
    _mm_free(input->id_nn);
    _mm_free(input->dist_nn);
    if (input->DS_quantized_plus) _mm_free(input->DS_quantized_plus);
    if (input->DS_quantized_minus) _mm_free(input->DS_quantized_minus);
    free(input);

    return 0;
}