; ============================================================================
; quantpivot64.nasm - Implementazione AVX 64-bit di euclidean_distance (FIXED)
; ============================================================================
; BUG FIX: I residui scalari vengono ora accumulati in un registro separato
; e sommati al totale PRIMA della riduzione orizzontale
; ============================================================================

default rel

section .text
global euclidean_distance_asm

euclidean_distance_asm:
    ; Inizializza accumulatore vettoriale a zero
    vxorpd ymm0, ymm0, ymm0        ; ymm0 = [0.0, 0.0, 0.0, 0.0] (somma vettoriale)
    vxorpd xmm2, xmm2, xmm2        ; xmm2 = 0.0 (somma scalare residui)
    
    ; Calcola numero di iterazioni vettoriali (D / 4)
    mov eax, edx                    ; eax = D
    shr eax, 2                      ; eax = D / 4
    jz .residual                    ; Se D < 4, salta al residuo

.vector_loop:
    ; Carica 4 double da v e w (32 byte)
    vmovupd ymm1, [rdi]             ; ymm1 = v[i..i+3]
    vmovupd ymm3, [rsi]             ; ymm3 = w[i..i+3]
    
    ; Calcola differenza
    vsubpd ymm1, ymm1, ymm3         ; ymm1 = v[i..i+3] - w[i..i+3]
    
    ; Eleva al quadrato
    vmulpd ymm1, ymm1, ymm1         ; ymm1 = diff²
    
    ; Accumula
    vaddpd ymm0, ymm0, ymm1         ; ymm0 += diff²
    
    ; Avanza puntatori
    add rdi, 32
    add rsi, 32
    
    ; Decrementa e ripeti
    dec eax
    jnz .vector_loop

.residual:
    ; Gestisci elementi residui (D mod 4)
    mov eax, edx                    ; eax = D
    and eax, 3                      ; eax = D mod 4
    jz .horizontal_sum              ; Se nessun residuo, vai alla somma

.residual_loop:
    ; Processa 1 double alla volta (scalare)
    vmovsd xmm1, [rdi]              ; xmm1 = v[i]
    vmovsd xmm3, [rsi]              ; xmm3 = w[i]
    
    vsubsd xmm1, xmm1, xmm3         ; xmm1 = v[i] - w[i]
    vmulsd xmm1, xmm1, xmm1         ; xmm1 = (v[i] - w[i])²
    vaddsd xmm2, xmm2, xmm1         ; xmm2 += (v[i] - w[i])² (ACCUMULA IN XMM2!)
    
    ; Avanza puntatori
    add rdi, 8
    add rsi, 8
    
    ; Decrementa e ripeti
    dec eax
    jnz .residual_loop

.horizontal_sum:
    ; Somma orizzontale dei 4 double in YMM0
    vextractf128 xmm1, ymm0, 1      ; xmm1 = [c, d] (parte alta)
    vaddpd xmm0, xmm0, xmm1         ; xmm0 = [a+c, b+d]
    vhaddpd xmm0, xmm0, xmm0        ; xmm0 = [a+b+c+d, a+b+c+d]
    
    ; ✅ FIX: Aggiungi la somma scalare dei residui
    vaddsd xmm0, xmm0, xmm2         ; xmm0 += somma_residui
    
    ; Calcola radice quadrata
    vsqrtsd xmm0, xmm0, xmm0        ; xmm0 = sqrt(sum)
    
    ; Cleanup
    vzeroupper
    ret