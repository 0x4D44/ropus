//! ropus-harness lib: hosts the FFI bindings so each bin can `use` them
//! instead of each pulling in a fresh copy via `#[path]`.
//!
//! The bindings module links against the xiph/opus C reference compiled
//! by `build.rs`. When that build step can't find `reference/`, it sets
//! `cfg(no_reference)` and this crate compiles to nothing — each
//! FFI-using binary stubs its `main` behind the same cfg. This keeps
//! `cargo build` at the workspace root succeeding on a fresh clone.

#[cfg(not(no_reference))]
pub mod bindings;

pub mod wav;

#[cfg(test)]
#[path = "../reference_build_manifest.rs"]
mod reference_build_manifest;

#[cfg(test)]
#[path = "../source_block_fingerprint.rs"]
mod source_block_fingerprint;

#[cfg(all(test, not(no_reference)))]
mod tests {
    use super::bindings;
    use ropus::celt::lpc::celt_fir;

    #[test]
    fn celt_fir_extreme_records_upstream_architecture_difference() {
        let c_input = [0_i16, 0, 0, -32768, -32768, -32768, -32768];
        let c_coefficients = [0_i16; 3];
        let mut c_output = [0_i16; 4];
        unsafe {
            bindings::debug_c_celt_fir(
                c_input.as_ptr(),
                c_coefficients.as_ptr(),
                c_output.as_mut_ptr(),
                c_output.len() as i32,
                c_coefficients.len() as i32,
            );
        }

        let rust_input = c_input.map(i32::from);
        let rust_coefficients = c_coefficients.map(i32::from);
        let mut rust_output = [0_i32; 4];
        celt_fir(
            &rust_input,
            &rust_coefficients,
            &mut rust_output,
            rust_input.len() - rust_coefficients.len(),
            rust_coefficients.len(),
        );

        assert_eq!(rust_output, [-32768; 4]);
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        assert_eq!(
            c_output, [-32767; 4],
            "upstream scalar SROUND16 has a narrower negative bound"
        );
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert_eq!(
            c_output, [-32768; 4],
            "upstream x86 SSE4.1 matches ropus's portable output"
        );
    }
}
