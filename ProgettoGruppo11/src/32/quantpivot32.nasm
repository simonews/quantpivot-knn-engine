
; Istruzioni: SSE/SSE2/SSE3 (128-bit registers)


; FIX RISPETTO ALLA VERSIONE PRECEDENTE:
; - Bug residui: usa XMM7 separato per accumulo scalare
; - Somma orizzontale: haddps (SSE3) invece di FPU stack


section .text
global euclidean_distance_asm


; float euclidean_distance_asm(const float* v, const float* w, int D)

; Parametri (cdecl - stack):
;   [ebp+8]  = const float* v   (primo parametro)
;   [ebp+12] = const float* w   (secondo parametro)
;   [ebp+16] = int D            (terzo parametro)
;
; Return:
;   ST(0) = float (FPU stack - convenzione 32-bit per float return)
;
; Registri salvati (callee-saved):
;   ebx, esi, edi, ebp

; INGRESSO FUNZIONE 

euclidean_distance_asm:
    
    ; FASE 0: salva registri e setup stack frame

    push ebp ; base pointer del chiamante 
    mov ebp, esp ; gli impostiamo l'attuale cima dello stack
    ; salva i registri 
    push ebx 
    push esi
    push edi
    
    
    ; FASE 1: Inizializzazione
    ; legge i parametri dallo stack

    mov esi, [ebp+8]            ; ESI = puntatore a v
    mov edi, [ebp+12]           ; EDI = puntatore a w
    mov ecx, [ebp+16]           ; ECX = D
    
    xorps xmm0, xmm0            ; XMM0 = [0.0, 0.0, 0.0, 0.0] (somma vettoriale)
    xorps xmm7, xmm7            ; XMM7 = 0.0 (somma residui scalari)
    
    
    ; FASE 2: Loop vettoriale (processa 4 float per iterazione)
    ; calcola quante iterazioni "piene" da 4 elementi possiamo fare

    mov eax, ecx                ; EAX = D
    shr eax, 2                  ; EAX = D / 4
    jz .residual                ; Se D < 4, salta al loop residuo
    
.vector_loop:
    ; Carica 4 float da v e w (16 byte = 4 × 4 byte)
    movups xmm1, [esi]          ; XMM1 = v[i..i+3]
    movups xmm2, [edi]          ; XMM2 = w[i..i+3]
    
    ; Calcola differenza: diff = v - w
    subps xmm1, xmm2            ; XMM1 = v[i..i+3] - w[i..i+3]
    
    ; Eleva al quadrato: diff²
    mulps xmm1, xmm1            ; XMM1 = (v[i] - w[i])^2
    
    ; Accumula nel sommatore
    addps xmm0, xmm1            ; XMM0 += diff²
    
    ; Avanza puntatori di 16 byte (4 float)
    add esi, 16
    add edi, 16
    
    ; Decrementa contatore e ripeti
    dec eax
    jnz .vector_loop
    
    
    ; FASE 3: Loop residuo (processa D mod 4 elementi scalari)
    
.residual:
    ; Calcola numero di elementi residui: eax = D mod 4
    mov eax, ecx                ; EAX = D
    and eax, 3                  ; EAX = D & 0b11 = D mod 4
    jz .horizontal_sum          ; Se nessun residuo, vai alla somma
    
.residual_loop:
    ; Processa 1 float alla volta (operazioni scalari)
    movss xmm1, [esi]           ; XMM1 = v[i]
    movss xmm2, [edi]           ; XMM2 = w[i]
    
    subss xmm1, xmm2            ; XMM1 = v[i] - w[i]
    mulss xmm1, xmm1            ; XMM1 = (v[i] - w[i])^2
    addss xmm7, xmm1            ; Accumulo scalare su XMM7 
    
    ; Avanza puntatori di 4 byte (1 float)
    add esi, 4
    add edi, 4
    
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
    
    
    ; FASE 6: Return in FPU stack (convenzione 32-bit per float)
    
    ; Nelle architetture x86 a 32 bit i valori di ritorno float o double NON vengono
    ; restituiti nei registri xmm0 ma nel registro ST(0)
    sub esp, 4                  ; Alloca spazio su stack
    movss [esp], xmm0           ; Salva risultato su stack
    fld dword [esp]             ; Carica dalla memoria allo stack FPU
    add esp, 4                  ; Pulisci stack
    
    
    ; FASE 7: ripristina registri e return

    pop edi
    pop esi
    pop ebx
    pop ebp
    ret