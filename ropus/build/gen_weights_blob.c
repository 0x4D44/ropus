/* Custom driver for producing `weights_blob.bin` from xiph's compile-time
   weight tables — strictly for ropus's build.rs to invoke.

   Derived from reference/dnn/write_lpcnet_weights.c, but narrowed to the
   three models ropus actually consumes at runtime (PitchDNN, FARGAN,
   PLC-predictor). Skipping DRED / OSCE keeps the blob small and avoids
   dragging in tables we don't yet port (~12 MB combined).

   Compile with `-DDUMP_BINARY_WEIGHTS` so the xiph `*_data.c` files omit
   their `init_*` bodies — we only need the `*_arrays[]` tables, not any
   of the layer-setup code. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nnet.h"

/* Include paths are wired through ropus/build.rs:
     reference/include     (for opus_types.h)
     reference/celt        (for os_support.h via arch.h, if pulled in)
     reference/dnn         (for nnet.h, *_data.h, and the *_data.c files below)
*/

/* The *_data.c files reference `opus_alloc`/`opus_free` indirectly via
   nnet.h includes, but with DUMP_BINARY_WEIGHTS set the only surviving
   symbols are the static data arrays + the WeightArray tables. No
   function calls we need to link against. */

#include "pitchdnn_data.c"
#include "fargan_data.c"
#include "plc_data.c"

/* Mirror of `write_weights` from reference/dnn/write_lpcnet_weights.c.
   Emits one WeightHead header (64 bytes) followed by `size` bytes of
   payload, padded with zeros so the next header starts on a 64-byte
   boundary. Matches the runtime parser in ropus::dnn::core::parse_weights. */
static void write_weights_table(const WeightArray *list, FILE *fout) {
    unsigned char zeros[WEIGHT_BLOCK_SIZE] = {0};
    int i = 0;
    while (list[i].name != NULL) {
        WeightHead h;
        memcpy(h.head, "DNNw", 4);
        h.version = WEIGHT_BLOB_VERSION;
        h.type = list[i].type;
        h.size = list[i].size;
        h.block_size =
            (h.size + WEIGHT_BLOCK_SIZE - 1) / WEIGHT_BLOCK_SIZE * WEIGHT_BLOCK_SIZE;
        memset(h.name, 0, sizeof(h.name));
        strncpy(h.name, list[i].name, sizeof(h.name) - 1);
        h.name[sizeof(h.name) - 1] = 0;
        fwrite(&h, 1, WEIGHT_BLOCK_SIZE, fout);
        fwrite(list[i].data, 1, h.size, fout);
        fwrite(zeros, 1, h.block_size - h.size, fout);
        i++;
    }
}

int main(int argc, char **argv) {
    const char *out_path =
        (argc >= 2) ? argv[1] : "weights_blob.bin";
    FILE *fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "gen_weights_blob: cannot open %s for writing\n", out_path);
        return 1;
    }
    write_weights_table(pitchdnn_arrays, fout);
    write_weights_table(fargan_arrays, fout);
    write_weights_table(plcmodel_arrays, fout);
    fclose(fout);
    return 0;
}
