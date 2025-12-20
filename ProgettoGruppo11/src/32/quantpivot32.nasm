section .data
align 16
temp_buffer: dd 0, 0, 0, 0

section .text
global euclidean_distance_asm

euclidean_distance_asm:
    push ebp
    mov ebp, esp
    push ebx
    push esi
    push edi
    
    mov esi, [ebp+8]        ; v
    mov edi, [ebp+12]       ; w
    mov ecx, [ebp+16]       ; D
    
    xorps xmm0, xmm0        ; somma vettoriale = 0
    
    ; Loop vettoriale
    mov eax, ecx
    shr eax, 2
    jz .residual
    
.vector_loop:
    movups xmm1, [esi]
    movups xmm2, [edi]
    subps xmm1, xmm2
    mulps xmm1, xmm1
    addps xmm0, xmm1
    
    add esi, 16
    add edi, 16
    dec eax
    jnz .vector_loop
    
.residual:
    mov eax, ecx
    and eax, 3
    jz .sum_elements
    
.residual_loop:
    movss xmm1, [esi]
    movss xmm2, [edi]
    subss xmm1, xmm2
    mulss xmm1, xmm1
    addss xmm0, xmm1
    
    add esi, 4
    add edi, 4
    dec eax
    jnz .residual_loop
    
.sum_elements:
    ; Salva xmm0 nel buffer globale
    movaps [temp_buffer], xmm0
    
    ; Somma con FPU
    fld dword [temp_buffer]
    fld dword [temp_buffer+4]
    faddp
    fld dword [temp_buffer+8]
    faddp
    fld dword [temp_buffer+12]
    faddp
    
    fsqrt
    
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret
