pub(crate) fn source_block_fnv1a64(
    source: &str,
    start_marker: &str,
    end_marker: &str,
) -> Result<u64, String> {
    let normalized = source.replace("\r\n", "\n");
    let start = normalized
        .find(start_marker)
        .ok_or_else(|| format!("missing source-block start marker: {start_marker}"))?;
    let relative_end = normalized[start..]
        .find(end_marker)
        .ok_or_else(|| format!("missing source-block end marker: {end_marker}"))?;
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in normalized[start..start + relative_end].bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::source_block_fnv1a64;

    #[test]
    fn fingerprint_is_line_ending_stable() {
        let unix = "before\nSTART\nbody\nEND\nafter";
        let windows = unix.replace('\n', "\r\n");
        assert_eq!(
            source_block_fnv1a64(unix, "START", "\nEND").unwrap(),
            source_block_fnv1a64(&windows, "START", "\nEND").unwrap()
        );
    }

    #[test]
    fn fingerprint_rejects_missing_markers() {
        assert!(source_block_fnv1a64("body", "START", "END").is_err());
    }
}
