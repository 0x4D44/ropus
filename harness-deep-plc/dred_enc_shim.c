/* Stage 8.4 C shim: expose the xiph C `dred_rdovae_encode_dframe` entry
 * point through a flat pointer API so we can call it from Rust tests
 * without replicating the `RDOVAEEnc` / `RDOVAEEncStruct` layouts across
 * the FFI boundary. Paired with `harness-deep-plc/src/lib.rs` bindings.
 *
 * The shim owns the struct allocations (malloc/free). Rust treats both
 * handles as opaque `void*`. */

#include <stdlib.h>
#include <string.h>

#include "dred_rdovae_enc.h"
#include "dred_rdovae_enc_data.h"
#include "dred_rdovae_constants.h"
#include "nnet.h"

/* Defined by `dred_rdovae_enc_data.c` (array of named weight descriptors
 * terminated by {NULL, 0, 0, NULL}). */
extern const WeightArray rdovaeenc_arrays[];

/* Allocate a fresh RDOVAE encoder model, initialise from the compiled-in
 * weight arrays. Returns NULL on alloc / init failure. Caller frees via
 * `ropus_test_rdovaeenc_free`. */
void *ropus_test_rdovaeenc_new(void) {
    RDOVAEEnc *model = (RDOVAEEnc *)calloc(1, sizeof(*model));
    if (!model) return NULL;
    if (init_rdovaeenc(model, rdovaeenc_arrays) != 0) {
        free(model);
        return NULL;
    }
    return model;
}

void ropus_test_rdovaeenc_free(void *model) {
    free(model);
}

/* Allocate a zeroed RDOVAEEncStruct (running state). */
void *ropus_test_rdovae_enc_state_new(void) {
    RDOVAEEncState *st = (RDOVAEEncState *)calloc(1, sizeof(*st));
    return st;
}

void ropus_test_rdovae_enc_state_free(void *state) {
    free(state);
}

/* Thin wrapper around the C forward pass; fixed `arch = 0` (scalar). */
void ropus_test_dred_rdovae_encode_dframe(
    void *state,
    const void *model,
    float *latents,
    float *initial_state,
    const float *input
) {
    dred_rdovae_encode_dframe(
        (RDOVAEEncState *)state,
        (const RDOVAEEnc *)model,
        latents,
        initial_state,
        input,
        /* arch = */ 0
    );
}
