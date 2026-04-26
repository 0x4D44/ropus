#include <math.h>
#include <stdio.h>
int main(void) {
#ifdef M_PI
    printf("M_PI=%.20f\n", (double)M_PI);
#else
    printf("M_PI not defined by math.h\n");
#endif
    return 0;
}
