#include <stdio.h>
#include <xmmintrin.h>

typedef float type;
extern type euclidean_distance_asm(const type* v, const type* w, int D);

int main() {
    type v[4] = {1.0, 2.0, 3.0, 4.0};
    type w[4] = {5.0, 6.0, 7.0, 8.0};
    
    type result = euclidean_distance_asm(v, w, 4);
    
    printf("Result: %f\n", result);
    printf("Expected: 8.0\n");
    
    return 0;
}
