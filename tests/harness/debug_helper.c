/* Small helper to extract SILK encoder indices from the C reference encoder. */

#include "opus.h"
#include "opus_types.h"
#include "silk/fixed/structs_FIX.h"

/* Internal opus_encoder layout - first two i32 fields are offsets */
typedef struct {
    opus_int32 celt_enc_offset;
    opus_int32 silk_enc_offset;
} OpusEncoderOffsets;

#include <stdio.h>

void debug_dump_silk_indices(OpusEncoder *enc) {
    OpusEncoderOffsets *offsets = (OpusEncoderOffsets *)enc;
    silk_encoder *silk = (silk_encoder *)((char *)enc + offsets->silk_enc_offset);
    silk_encoder_state *state = &silk->state_Fxx[0].sCmn;

    fprintf(stderr, "[C SILK] speech_activity_Q8=%d signal_type=%d quant_offset=%d\n",
        state->speech_activity_Q8,
        (int)state->indices.signalType,
        (int)state->indices.quantOffsetType);
    fprintf(stderr, "[C SILK] gains_indices=[%d, %d, %d, %d] nb_subfr=%d\n",
        (int)state->indices.GainsIndices[0],
        (int)state->indices.GainsIndices[1],
        (int)state->indices.GainsIndices[2],
        (int)state->indices.GainsIndices[3],
        state->nb_subfr);
    fprintf(stderr, "[C SILK] nlsf_indices=[%d",
        (int)state->indices.NLSFIndices[0]);
    {
        int i;
        for (i = 1; i <= 10; i++) {
            fprintf(stderr, ", %d", (int)state->indices.NLSFIndices[i]);
        }
    }
    fprintf(stderr, "]\n");
    fprintf(stderr, "[C SILK] VAD_flags=[%d] LBRR_flag=%d inDTX=%d\n",
        (int)state->VAD_flags[0],
        (int)state->LBRR_flag,
        state->inDTX);
    fprintf(stderr, "[C SILK] seed=%d n_frames_encoded=%d\n",
        (int)state->indices.Seed,
        state->nFramesEncoded);
}
