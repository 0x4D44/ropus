/* Stage 8.5 C shim: expose the xiph C `dred_rdovae_decode_qframe` and
 * `dred_rdovae_dec_init_states` entry points through a flat pointer API
 * so we can call them from Rust tests without replicating the
 * `RDOVAEDec` / `RDOVAEDecStruct` layouts across the FFI boundary.
 * Paired with `harness-deep-plc/src/lib.rs` bindings.
 *
 * The shim owns the struct allocations (malloc/free). Rust treats both
 * handles as opaque `void*`. */

#include <stdlib.h>
#include <string.h>

#include "dred_rdovae_dec.h"
#include "dred_rdovae_dec_data.h"
#include "dred_rdovae_constants.h"
#include "nnet.h"

/* Defined by `dred_rdovae_dec_data.c` (array of named weight descriptors
 * terminated by {NULL, 0, 0, NULL}). */
extern const WeightArray rdovaedec_arrays[];

/* Allocate a fresh RDOVAE decoder model, initialise from the compiled-in
 * weight arrays. Returns NULL on alloc / init failure. Caller frees via
 * `ropus_test_rdovaedec_free`. */
void *ropus_test_rdovaedec_new(void) {
    RDOVAEDec *model = (RDOVAEDec *)calloc(1, sizeof(*model));
    if (!model) return NULL;
    if (init_rdovaedec(model, rdovaedec_arrays) != 0) {
        free(model);
        return NULL;
    }
    return model;
}

void ropus_test_rdovaedec_free(void *model) {
    free(model);
}

/* Allocate a zeroed RDOVAEDecStruct (running state). */
void *ropus_test_rdovae_dec_state_new(void) {
    RDOVAEDecState *st = (RDOVAEDecState *)calloc(1, sizeof(*st));
    return st;
}

void ropus_test_rdovae_dec_state_free(void *state) {
    free(state);
}

/* Thin wrapper around the C init helper; fixed `arch = 0` (scalar). */
void ropus_test_dred_rdovae_dec_init_states(
    void *state,
    const void *model,
    const float *initial_state
) {
    dred_rdovae_dec_init_states(
        (RDOVAEDecState *)state,
        (const RDOVAEDec *)model,
        initial_state,
        /* arch = */ 0
    );
}

/* Thin wrapper around the C forward pass; fixed `arch = 0` (scalar). */
void ropus_test_dred_rdovae_decode_qframe(
    void *state,
    const void *model,
    float *qframe,
    const float *input
) {
    dred_rdovae_decode_qframe(
        (RDOVAEDecState *)state,
        (const RDOVAEDec *)model,
        qframe,
        input,
        /* arch = */ 0
    );
}
