/* Silicon truth: does a PRIME leading dimension beat a POWER-OF-TWO one because
 * power-of-two strides alias into the same cache sets (conflict misses)?
 * 6-neighbour 3D stencil (the gauge-link update), uint8, same interior work. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double bench(int W, int H, int Dp, int reps) {
    size_t n = (size_t)W * H * Dp;
    uint8_t *g = malloc(n), *o = malloc(n);
    for (size_t t = 0; t < n; t++) g[t] = (uint8_t)(t * 7 + 1);
    /* strides: i contiguous, j -> W, k -> W*H (this is the aliasing one) */
    size_t sj = W, sk = (size_t)W * H;
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    volatile uint64_t sink = 0;
    for (int r = 0; r < reps; r++) {
        for (int k = 1; k < Dp - 1; k++)
        for (int j = 1; j < H - 1; j++)
        for (int i = 1; i < W - 1; i++) {
            size_t c = (size_t)k * sk + (size_t)j * sj + i;
            o[c] = (uint8_t)(g[c-1] + g[c+1] + g[c-sj] + g[c+sj] + g[c-sk] + g[c+sk]);
        }
        sink += o[(n >> 1) | 1];
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
    double nodes = (double)(W-2)*(H-2)*(Dp-2)*reps;
    free(g); free(o);
    (void)sink;
    return sec/nodes*1e9;   /* ns per node */
}

int main(void) {
    int reps = 40;
    /* isolate the leading dim: same H,D; W = 128 (pow2) vs 127 (prime) vs 113 (prime) */
    printf("pow2  W=128 H=128 D=256 : %.3f ns/node\n", bench(128,128,256,reps));
    printf("prime W=127 H=128 D=256 : %.3f ns/node\n", bench(127,128,256,reps));
    printf("prime W=113 H=128 D=256 : %.3f ns/node\n", bench(113,128,256,reps));
    /* the paper's manifold vs traditional (different aspect, normalized per node) */
    printf("traditional 128x128x256 : %.3f ns/node\n", bench(128,128,256,reps));
    printf("manifold    157x64 x256 : %.3f ns/node\n", bench(157,64,256,reps));
    return 0;
}
