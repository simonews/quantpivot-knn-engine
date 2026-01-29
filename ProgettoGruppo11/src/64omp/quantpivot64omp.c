#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include "common.h"
#include <stdint.h>


extern type euclidean_distance_asm(const type* v, const type* w, int D);


typedef struct {
    type abs_val;
    int idx;
} pair_t;

// funzione di comparazione per qsort
int compare_pairs(const void* a, const void* b) {
    pair_t* pa = (pair_t*)a;
    pair_t* pb = (pair_t*)b;
    
    // Ordinamento decrescente per valore assoluto
    if (pa->abs_val > pb->abs_val) return -1;
    if (pa->abs_val < pb->abs_val) return 1;
    return 0;
}


// QUANTIZE - Trasforma vettore in rappresentazione binaria sparsa
void quantize(const type* v, int D, int x, 
              uint8_t* v_plus, uint8_t* v_minus) {
    
    // Crea array di coppie (|v[i]|, indice)
    pair_t* pairs = malloc(D * sizeof(pair_t));
    if (!pairs) {
        fprintf(stderr, "Errore allocazione in quantize\n");
        exit(1);
    }
    
    for (int i = 0; i < D; i++) {
        pairs[i].abs_val = fabs(v[i]);
        pairs[i].idx = i;
    }
    
    // Trova x elementi con valore assoluto massimo
    qsort(pairs, D, sizeof(pair_t), compare_pairs);
    
    // Inizializza v_plus e v_minus a 0
    memset(v_plus, 0, D * sizeof(uint8_t));
    memset(v_minus, 0, D * sizeof(uint8_t));
    
    // Setta bit per i primi x elementi
    for (int j = 0; j < x && j < D; j++) {
        int idx = pairs[j].idx;
        if (v[idx] >= 0) {
            v_plus[idx] = 1;
        } else {
            v_minus[idx] = 1;
        }
    }
    
    free(pairs);
}

// APPROX_DISTANCE - Distanza approssimata tra vettori quantizzati
type approx_distance(const uint8_t* vp, const uint8_t* vm,
                     const uint8_t* wp, const uint8_t* wm, int D) {
    int dot_pp = 0, dot_mm = 0, dot_pm = 0, dot_mp = 0;
    
    for (int i = 0; i < D; i++) {
        // Prodotto scalare binario = AND bit a bit
        dot_pp += (vp[i] & wp[i]);
        dot_mm += (vm[i] & wm[i]);
        dot_pm += (vp[i] & wm[i]);
        dot_mp += (vm[i] & wp[i]);
    }
    
    // Formula: (v+·w+) + (v-·w-) - (v+·w-) - (v-·w+)
    return (type)(dot_pp + dot_mm - dot_pm - dot_mp);
}


// EUCLIDEAN_DISTANCE - Distanza euclidea esatta (versione C)
type euclidean_distance_c(const type* v, const type* w, int D) {
    type sum = 0.0;
    
    for (int i = 0; i < D; i++) {
        type diff = v[i] - w[i];
        sum += diff * diff;
    }
    
    return sqrt(sum);
}


// EUCLIDEAN_DISTANCE - wrapper
type euclidean_distance(const type* v, const type* w, int D) {
    
    //return euclidean_distance_c(v, w, D);
    return euclidean_distance_asm(v, w, D);  
}



// FIT - Costruzione dell'indice (PARALLELIZZATO)
void fit(params* input) {
    if (!input->silent) {
        printf("[FIT] Inizio costruzione indice...\n");
        printf("      N=%d, D=%d, h=%d, x=%d\n", 
               input->N, input->D, input->h, input->x);
    }
    
    // Verifica parametri
    if (input->N < input->h) {
        fprintf(stderr, "Errore: N < h\n");
        exit(1);
    }
    
    // Alloca array pivot (solo indici)
    input->P = _mm_malloc(input->h * sizeof(int), align);
    if (!input->P) {
        fprintf(stderr, "Errore allocazione pivot\n");
        exit(1);
    }
    
    // Seleziona h pivot (campionamento uniforme)
    int step = input->N / input->h;
    if (!input->silent) printf("[FIT] Selezione pivot (step=%d)...\n", step);
    
    for (int j = 0; j < input->h; j++) {
        int pivot_idx = step * j;
        if (pivot_idx >= input->N) pivot_idx = input->N - 1;
        input->P[j] = pivot_idx;
    }
    
    // Alloca indice [N x h]
    input->index = _mm_malloc(input->N * input->h * sizeof(type), align);
    if (!input->index) {
        fprintf(stderr, "Errore allocazione indice\n");
        exit(1);
    }
    
    // Alloca array per vettori quantizzati
    if (!input->silent) printf("[FIT] Allocazione vettori quantizzati...\n");
    uint8_t* DS_vp = malloc(input->N * input->D * sizeof(uint8_t));
    uint8_t* DS_vm = malloc(input->N * input->D * sizeof(uint8_t));
    uint8_t* P_vp = malloc(input->h * input->D * sizeof(uint8_t));
    uint8_t* P_vm = malloc(input->h * input->D * sizeof(uint8_t));
    
    if (!DS_vp || !DS_vm || !P_vp || !P_vm) {
        fprintf(stderr, "Errore allocazione in fit\n");
        exit(1);
    }
    
    // Quantizza tutti i punti del dataset (PARALLELIZZATO)
    if (!input->silent) printf("[FIT] Quantizzazione dataset (%d punti)...\n", input->N);
    // schedule(static): costo uniforme per ogni iterazione
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < input->N; i++) {
        quantize(&input->DS[i * input->D], input->D, input->x,
                 &DS_vp[i * input->D], &DS_vm[i * input->D]);
    }
    
    // Quantizza tutti i pivot (sequenziale, h piccolo)
    if (!input->silent) printf("[FIT] Quantizzazione pivot (%d pivot)...\n", input->h);
    for (int j = 0; j < input->h; j++) {
        int pivot_idx = input->P[j];
        quantize(&input->DS[pivot_idx * input->D], input->D, input->x,
                 &P_vp[j * input->D], &P_vm[j * input->D]);
    }
    
    // Costruisci indice: per ogni punto DS, calcola distanza a ogni pivot (PARALLELIZZATO)
    if (!input->silent) printf("[FIT] Costruzione indice [%d x %d]...\n", input->N, input->h);
    // collapse(2): unisce i due loop per bilanciare meglio il carico
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < input->N; i++) {
        for (int j = 0; j < input->h; j++) {
            input->index[i * input->h + j] = 
                approx_distance(&DS_vp[i * input->D], &DS_vm[i * input->D],
                               &P_vp[j * input->D], &P_vm[j * input->D],
                               input->D);
        }
    }
    
    // SALVA dataset quantizzato per predict
    input->DS_quantized_plus = _mm_malloc(input->N * input->D * sizeof(uint8_t), align);
    input->DS_quantized_minus = _mm_malloc(input->N * input->D * sizeof(uint8_t), align);
    memcpy(input->DS_quantized_plus, DS_vp, input->N * input->D * sizeof(uint8_t));
    memcpy(input->DS_quantized_minus, DS_vm, input->N * input->D * sizeof(uint8_t));
    
    // Cleanup
    free(DS_vp); 
    free(DS_vm);
    free(P_vp); 
    free(P_vm);
    
    if (!input->silent) printf("[FIT] Completato!\n");
}


// PREDICT - Ricerca K-NN con pruning (PARALLELIZZATO)
void predict(params* input) {
    if (!input->silent) {
        printf("[PREDICT] Inizio ricerca K-NN...\n");
        printf("          nq=%d, k=%d\n", input->nq, input->k);
    }
    
    // Quantizza pivot (sequenziale, h piccolo)
    uint8_t* P_vp = malloc(input->h * input->D * sizeof(uint8_t));
    uint8_t* P_vm = malloc(input->h * input->D * sizeof(uint8_t));
    
    for (int j = 0; j < input->h; j++) {
        int pivot_idx = input->P[j];
        quantize(&input->DS[pivot_idx * input->D], input->D, input->x,
                 &P_vp[j * input->D], &P_vm[j * input->D]);
    }
    
    // Usa dataset pre-quantizzato da fit()
    uint8_t* DS_vp = input->DS_quantized_plus;
    uint8_t* DS_vm = input->DS_quantized_minus;
        
    // PARALLELIZZAZIONE su query (schedule(dynamic) per pruning disuguale)
    // Ogni thread ha i propri buffer privati
    #pragma omp parallel
    {
        /*
        *  Alllocazione dentro la regione parallela -> ogni thread allora i 
        *  suoi buffer personali, altrimenti scriverebbero tutti nello stesso buffer
        */
        uint8_t* q_vp = malloc(input->D * sizeof(uint8_t));
        uint8_t* q_vm = malloc(input->D * sizeof(uint8_t));
        type* q_to_pivots = malloc(input->h * sizeof(type));
        int* knn_ids = malloc(input->k * sizeof(int));
        type* knn_dists = malloc(input->k * sizeof(type));
        
        // Loop parallelo sulle query (schedule dinamico qui)
        #pragma omp for schedule(dynamic)
        for (int qi = 0; qi < input->nq; qi++) {
            
            if (!input->silent && ((qi + 1) % 100 == 0 || qi == 0)) {
                // printf in parallelo può sovrapporsi ma è accettabile per debug
                #pragma omp critical
                {
                    printf(" Query %d/%d (thread %d)\n", qi+1, input->nq, omp_get_thread_num());
                }
            }
            
            type* q = &input->Q[qi * input->D];
            
            // Quantizza query
            quantize(q, input->D, input->x, q_vp, q_vm);
            
            // Calcola distanze query → pivot
            for (int j = 0; j < input->h; j++) {
                q_to_pivots[j] = approx_distance(q_vp, q_vm,
                                                 &P_vp[j * input->D],
                                                 &P_vm[j * input->D],
                                                 input->D);
            }
            
            // Inizializza lista K-NN
            for (int i = 0; i < input->k; i++) {
                knn_ids[i] = -1;
                knn_dists[i] = INFINITY;
            }
            
            // Scansione dataset con pruning
            int pruned = 0;
            for (int i = 0; i < input->N; i++) {
                // Calcola bound triangolare (max su tutti i pivot)
                type max_bound = 0.0;
                for (int j = 0; j < input->h; j++) {
                    type bound = fabs(input->index[i * input->h + j] - q_to_pivots[j]);
                    if (bound > max_bound) max_bound = bound;
                }
                
                // Pruning: se bound >= k-esimo vicino, skip
                type d_max_k = knn_dists[input->k - 1];
                if (max_bound >= d_max_k) {
                    pruned++;
                    continue;
                }
                
                // Calcola distanza approssimata effettiva
                type dist_approx = approx_distance(q_vp, q_vm,
                                                   &DS_vp[i * input->D],
                                                   &DS_vm[i * input->D],
                                                   input->D);
                
                // Se migliore del k-esimo, inserisci in lista ordinata
                if (dist_approx < d_max_k) {
                    // Trova posizione e shifta elementi
                    int pos = input->k - 1;
                    while (pos > 0 && dist_approx < knn_dists[pos - 1]) {
                        knn_dists[pos] = knn_dists[pos - 1];
                        knn_ids[pos] = knn_ids[pos - 1];
                        pos--;
                    }
                    knn_dists[pos] = dist_approx;
                    knn_ids[pos] = i;
                }
            }
            
            // Raffinamento: distanza euclidea esatta sui K candidati
            for (int idx = 0; idx < input->k; idx++) {
                if (knn_ids[idx] >= 0) {
                    knn_dists[idx] = euclidean_distance(q,
                                                        &input->DS[knn_ids[idx] * input->D],
                                                        input->D);
                }
            }
            
            // Riordina dopo raffinamento (bubble sort per k piccolo)
            for (int pass = 0; pass < input->k - 1; pass++) {
                for (int j = 0; j < input->k - 1 - pass; j++) {
                    if (knn_dists[j] > knn_dists[j + 1]) {
                        // Swap distanze
                        type tmp_d = knn_dists[j];
                        knn_dists[j] = knn_dists[j + 1];
                        knn_dists[j + 1] = tmp_d;
                        // Swap ID
                        int tmp_id = knn_ids[j];
                        knn_ids[j] = knn_ids[j + 1];
                        knn_ids[j + 1] = tmp_id;
                    }
                }
            }
            
            // Salva risultati (thread-safe: ogni thread ha un qi univoco grazie  a #pragma omp for)
            memcpy(&input->id_nn[qi * input->k], knn_ids, input->k * sizeof(int));
            memcpy(&input->dist_nn[qi * input->k], knn_dists, input->k * sizeof(type));
        }
        
        // Cleanup dei buffer 
        free(q_vp); 
        free(q_vm); 
        free(q_to_pivots);
        free(knn_ids); 
        free(knn_dists);
    }
    
    // Cleanup globale
    free(P_vp); 
    free(P_vm);
   
    if (!input->silent) printf("[PREDICT] Completato!\n");
}