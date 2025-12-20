%include "sseutils32.nasm"

section .data
    align 16

section .text

global euclidean_distance_asm

; extern float euclidean_distance_asm(const float* v, const float* w, int D);
; Calcola la distanza euclidea tra due vettori v e w di dimensione D
; Parametri:
;   ESP+4  = v (puntatore al primo vettore)
;   ESP+8  = w (puntatore al secondo vettore)
;   ESP+12 = D (dimensione dei vettori)
; Ritorna: distanza euclidea in XMM0

euclidean_distance_asm:
    push ebp
    mov ebp, esp
    push ebx
    push esi
    push edi
    
    ; Carica i parametri
    mov esi, [ebp+8]        ; esi = v
    mov edi, [ebp+12]       ; edi = w
    mov ecx, [ebp+16]       ; ecx = D
    
    ; Inizializza accumulatore della somma a zero
    xorps xmm0, xmm0        ; xmm0 = somma delle differenze al quadrato
    
    ; Calcola quanti gruppi di 4 elementi possiamo processare
    mov eax, ecx
    shr eax, 2              ; eax = D / 4 (parte vettoriale)
    test eax, eax
    jz .residual            ; Se D < 4, salta alla parte residua
    
    ; Loop vettoriale: processa 4 elementi alla volta
.vector_loop:
    movaps xmm1, [esi]      ; xmm1 = v[i..i+3]
    movaps xmm2, [edi]      ; xmm2 = w[i..i+3]
    
    subps xmm1, xmm2        ; xmm1 = v[i] - w[i] per i in [0..3]
    mulps xmm1, xmm1        ; xmm1 = (v[i] - w[i])^2
    
    addps xmm0, xmm1        ; xmm0 += differenze al quadrato
    
    add esi, 16             ; v += 4 elementi (4 * 4 bytes)
    add edi, 16             ; w += 4 elementi
    
    dec eax
    jnz .vector_loop
    
.residual:
    ; Gestisci gli elementi residui (D % 4)
    mov eax, ecx
    and eax, 3              ; eax = D % 4
    test eax, eax
    jz .horizontal_sum      ; Se non ci sono residui, vai alla somma
    
.residual_loop:
    movss xmm1, [esi]       ; xmm1 = v[i] (singolo float)
    movss xmm2, [edi]       ; xmm2 = w[i] (singolo float)
    
    subss xmm1, xmm2        ; xmm1 = v[i] - w[i]
    mulss xmm1, xmm1        ; xmm1 = (v[i] - w[i])^2
    
    addss xmm0, xmm1        ; xmm0 += differenza al quadrato
    
    add esi, 4              ; v++ (1 elemento = 4 bytes)
    add edi, 4              ; w++
    
    dec eax
    jnz .residual_loop
    
.horizontal_sum:
    ; Somma orizzontale dei 4 elementi di xmm0
    ; xmm0 = [a, b, c, d]
    
    movaps xmm1, xmm0       ; xmm1 = [a, b, c, d]
    shufps xmm1, xmm1, 0xB1 ; xmm1 = [b, a, d, c] (swap adjacent)
    addps xmm0, xmm1        ; xmm0 = [a+b, a+b, c+d, c+d]
    
    movaps xmm1, xmm0       ; xmm1 = [a+b, a+b, c+d, c+d]
    shufps xmm1, xmm1, 0x4E ; xmm1 = [c+d, c+d, a+b, a+b] (swap high/low)
    addps xmm0, xmm1        ; xmm0 = [a+b+c+d, *, *, *]
    
    ; Calcola la radice quadrata
    sqrtss xmm0, xmm0       ; xmm0 = sqrt(somma)
    
    ; Il risultato è già in xmm0 (convenzione di ritorno per float)
    
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret