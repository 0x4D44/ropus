//! Shared guards for differential oracles that compare neural f32 output.
//!
//! A non-finite value must fail before bit-exact or SNR logic sees it. In
//! particular, NaN makes ordered comparisons false and can leave a running
//! worst-case metric at its optimistic initial value.

pub fn finite_slice_error(label: &str, values: &[f32]) -> Option<String> {
    values.iter().enumerate().find_map(|(index, &value)| {
        (!value.is_finite())
            .then(|| format!("{label} contains non-finite value at index {index}: {value:?}"))
    })
}

pub fn finite_pair_error(label: &str, reference: &[f32], candidate: &[f32]) -> Option<String> {
    assert_eq!(
        reference.len(),
        candidate.len(),
        "{label} outputs must have equal lengths"
    );
    reference
        .iter()
        .zip(candidate)
        .enumerate()
        .find_map(|(index, (&reference_value, &candidate_value))| {
            if !reference_value.is_finite() {
                Some(format!(
                    "{label} reference contains non-finite value at index {index}: {reference_value:?}"
                ))
            } else if !candidate_value.is_finite() {
                Some(format!(
                    "{label} candidate contains non-finite value at index {index}: {candidate_value:?}"
                ))
            } else {
                None
            }
        })
}

pub fn assert_finite_slice(label: &str, values: &[f32]) {
    if let Some(error) = finite_slice_error(label, values) {
        panic!("{error}");
    }
}

pub fn assert_finite_pair(label: &str, reference: &[f32], candidate: &[f32]) {
    if let Some(error) = finite_pair_error(label, reference, candidate) {
        panic!("{error}");
    }
}

#[cfg(test)]
mod tests {
    use super::{finite_pair_error, finite_slice_error};

    #[test]
    fn finite_control_is_accepted() {
        assert_eq!(
            finite_pair_error("control", &[0.0, -1.25], &[2.0, 3.5]),
            None
        );
        assert_eq!(finite_slice_error("control", &[0.0, -1.25, 3.5]), None);
    }

    #[test]
    fn same_nan_is_rejected() {
        let nan = f32::NAN;
        let error = finite_pair_error("same NaN", &[nan], &[nan]).expect("NaN must fail");
        assert!(error.contains("reference"));
        assert!(error.contains("NaN"));
    }

    #[test]
    fn one_sided_nan_is_rejected() {
        let reference_error = finite_pair_error("reference NaN", &[f32::NAN], &[1.0])
            .expect("reference NaN must fail");
        let candidate_error = finite_pair_error("candidate NaN", &[1.0], &[f32::NAN])
            .expect("candidate NaN must fail");
        assert!(reference_error.contains("reference"));
        assert!(candidate_error.contains("candidate"));
    }

    #[test]
    fn positive_and_negative_infinity_are_rejected() {
        for value in [f32::INFINITY, f32::NEG_INFINITY] {
            assert!(finite_slice_error("infinity", &[value]).is_some());
            assert!(finite_pair_error("infinity", &[0.0], &[value]).is_some());
        }
    }
}
