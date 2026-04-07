// Stub: ambisonics mixing/demixing matrix data.
// TODO: generate from C reference tables (mapping_matrix_data.h).
// These empty arrays allow compilation; the projection encoder/decoder
// will fail at runtime if actually used with orders that reference them.

#[allow(dead_code)]
static FOA_MIXING_DATA: [i16; 36] = [0i16; 36]; // 6x6
#[allow(dead_code)]
static FOA_DEMIXING_DATA: [i16; 36] = [0i16; 36]; // 6x6
#[allow(dead_code)]
static SOA_MIXING_DATA: [i16; 121] = [0i16; 121]; // 11x11
#[allow(dead_code)]
static SOA_DEMIXING_DATA: [i16; 121] = [0i16; 121]; // 11x11
#[allow(dead_code)]
static TOA_MIXING_DATA: [i16; 324] = [0i16; 324]; // 18x18
#[allow(dead_code)]
static TOA_DEMIXING_DATA: [i16; 324] = [0i16; 324]; // 18x18
#[allow(dead_code)]
static FOURTHOA_MIXING_DATA: [i16; 729] = [0i16; 729]; // 27x27
#[allow(dead_code)]
static FOURTHOA_DEMIXING_DATA: [i16; 729] = [0i16; 729]; // 27x27
#[allow(dead_code)]
static FIFTHOA_MIXING_DATA: [i16; 1444] = [0i16; 1444]; // 38x38
#[allow(dead_code)]
static FIFTHOA_DEMIXING_DATA: [i16; 1444] = [0i16; 1444]; // 38x38
