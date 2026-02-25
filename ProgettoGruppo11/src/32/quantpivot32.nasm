

section .text
global euclidean_distance_asm


; Parametri:
;   RDI = const float* v   (primo parametro)
;   RSI = const float* w   (secondo parametro)
;   EDX = int D            (terzo parametro)
;
; Return: XMM0


euclidean_distance_asm:
    
    ; FASE 0: setup stack frame
    ; usa registri a 64 bit per push/pop 

    push rbp 
    mov rbp, rsp 
    
    
    
    
    ; FASE 1: Inizializzazione
    ; i parametri sono già in RDI, RSI, EDX

    
    xorps xmm0, xmm0            ; XMM0 = [0.0, 0.0, 0.0, 0.0] (somma vettoriale)
    xorps xmm7, xmm7            ; XMM7 = 0.0 (somma residui scalari)
    
    
    ; FASE 2: Loop vettoriale (processa 4 float per iterazione)
    ; calcola quante iterazioni "piene" da 4 elementi possiamo fare

    mov eax, edx                ; EAX = D
    shr eax, 2                  ; EAX = D / 4
    jz .residual                ; Se D < 4, salta al loop residuo
    
.vector_loop:
    ; Carica 4 float da v e w (16 byte = 4 × 4 byte)
    movups xmm1, [rdi]          ; carica puntatore RDI
    movups xmm2, [rsi]          ; carica puntatore RSI
    
    ; Calcola differenza: diff = v - w
    subps xmm1, xmm2            ; 
    
    ; Eleva al quadrato: diff^2
    mulps xmm1, xmm1            ; XMM1 = (v[i] - w[i])^2
    
    ; Accumula nel sommatore
    addps xmm0, xmm1            ; XMM0 += diff^2
    
    ; Avanza puntatori di 16 byte (4 float)
    add rdi, 16
    add rsi, 16
    
    ; Decrementa contatore e ripeti
    dec eax
    jnz .vector_loop
    
    
    ; FASE 3: Loop residuo (processa D mod 4 elementi scalari)
    
.residual:
    ; Calcola numero di elementi residui: eax = D mod 4
    mov eax, edx                ; EAX = D
    and eax, 3                  ; EAX = D & 0b11 = D mod 4
    jz .horizontal_sum          ; Se nessun residuo, vai alla somma
    
.residual_loop:
    ; Processa 1 float alla volta (operazioni scalari)
    movss xmm1, [rdi]           ; XMM1 = v[i]
    movss xmm2, [rsi]           ; XMM2 = w[i]
    
    subss xmm1, xmm2            ; XMM1 = v[i] - w[i]
    mulss xmm1, xmm1            ; XMM1 = (v[i] - w[i])^2
    addss xmm7, xmm1            ; Accumulo scalare su XMM7 
    
    ; Avanza puntatori di 4 byte (1 float)
    add rdi, 4
    add rsi, 4
    
    ; Decrementa e ripeti
    dec eax
    jnz .residual_loop
    
    
    ; FASE 4: Somma orizzontale 
    ; dobbiamo trasformare le 4 somme parziali in XMM0 in un unico numero 
    
.horizontal_sum:
    
    
    ; Somma orizzontale con haddps
    movaps xmm1, xmm0           ; copia di XMM0 in XMM1 = [a, b, c, d]
    haddps xmm0, xmm1           ; XMM0 = [a+b, c+d, a+b, c+d]
    haddps xmm0, xmm0           ; XMM0 = [a+b+c+d, *, *, *]
    ; ora il primo elemento (bit 0-31) contiene la somma totale della parte vettoriale 
    
    ; Aggiungi la somma scalare dei residui
    addss xmm0, xmm7            ; XMM0[0] += somma_residui
    
    
    ; FASE 5: Radice quadrata
    
    sqrtss xmm0, xmm0           ; XMM0 = sqrt(sum)
    
    
    ; FASE 6 e 7: ripristina e returna
    ; nelle architetture a 64 bit returniamo in XMM0

    pop rbp
    ret
