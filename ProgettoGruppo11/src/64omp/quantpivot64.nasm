section .text
global euclidean_distance_asm

euclidean_distance_asm:
    ; Calling convention System V AMD64:
    ; RDI = const double* v
    ; RSI = const double* w
    ; EDX = int D
    ; Return: double in XMM0

    ; Inizializza accumulatore a zero
    vxorpd ymm0, ymm0, ymm0        ; ymm0 = [0.0, 0.0, 0.0, 0.0]

    ; Calcola numero di iterazioni vettoriali (D / 4)
    mov eax, edx                    ; eax = D
    shr eax, 2                      ; eax = D / 4 (shift right di 2 bit)
    jz .residual                    ; Se D < 4, salta al residuo

.vector_loop:
    ; Carica 4 double da v e w (32 byte = 4 × 8 byte)
    vmovupd ymm1, [rdi]             ; ymm1 = v[i..i+3]
    vmovupd ymm2, [rsi]             ; ymm2 = w[i..i+3]

    ; Calcola differenza: diff = v - w
    vsubpd ymm1, ymm1, ymm2         ; ymm1 = v[i..i+3] - w[i..i+3]

    ; Eleva al quadrato: diff² = diff * diff
    vmulpd ymm1, ymm1, ymm1         ; ymm1 = diff²

    ; Accumula nel sommatore
    vaddpd ymm0, ymm0, ymm1         ; ymm0 += diff²

    ; Avanza puntatori di 32 byte (4 double)
    add rdi, 32
    add rsi, 32

    ; Decrementa contatore e ripeti
    dec eax
    jnz .vector_loop

.residual:
    ; Gestisci elementi residui (D mod 4)
    mov eax, edx                    ; eax = D
    and eax, 3                      ; eax = D mod 4 (maschera ultimi 2 bit)
    jz .horizontal_sum              ; Se nessun residuo, vai alla somma

.residual_loop:
    ; Processa 1 double alla volta (scalare)
    vmovsd xmm1, [rdi]              ; xmm1 = v[i]
    vmovsd xmm2, [rsi]              ; xmm2 = w[i]
    vsubsd xmm1, xmm1, xmm2         ; xmm1 = v[i] - w[i]
    vmulsd xmm1, xmm1, xmm1         ; xmm1 = (v[i] - w[i])²
    vaddsd xmm0, xmm0, xmm1         ; xmm0 += (v[i] - w[i])²

    ; Avanza puntatori di 8 byte (1 double)
    add rdi, 8
    add rsi, 8

    ; Decrementa e ripeti
    dec eax
    jnz .residual_loop

.horizontal_sum:
    ; Somma orizzontale dei 4 double in YMM0
    ; YMM0 = [a, b, c, d] → XMM0 = a+b+c+d

    ; Estrai parte alta (128-bit superiori) in XMM1
    vextractf128 xmm1, ymm0, 1      ; xmm1 = [c, d]

    ; Somma parte bassa e alta
    vaddpd xmm0, xmm0, xmm1         ; xmm0 = [a+c, b+d]

    ; Somma i 2 double rimanenti
    vhaddpd xmm0, xmm0, xmm0        ; xmm0 = [a+b+c+d, a+b+c+d]

    ; Calcola radice quadrata
    vsqrtsd xmm0, xmm0, xmm0        ; xmm0 = sqrt(sum)

    ; Pulisci stato AVX prima del return
    vzeroupper                       ; ✅ CORRETTO (non 'vzerouppper')

    ret