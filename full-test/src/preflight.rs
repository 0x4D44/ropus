//! Release preflight asset probes.
//!
//! `--release-preflight` has two claim profiles. The quick profile is a core
//! smoke gate and does not claim neural/DRED coverage; the non-quick profile
//! includes the deep-PLC/DRED package lane and requires its assets. IETF vector
//! absence is shown here, but the existing Stage 2 synthetic conformance failure
//! remains the only failure accounting path for vectors so totals are not
//! double-counted.

use std::path::{Path, PathBuf};

use crate::ietf_vectors::{IetfVectorProvision, ProvisionStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetRequirement {
    Required,
    Optional,
}

impl AssetRequirement {
    pub fn as_str(self) -> &'static str {
        match self {
            AssetRequirement::Required => "required",
            AssetRequirement::Optional => "optional",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightPolicy {
    DefaultReportOnly,
    ReleaseCoreSmokeNoNeuralClaim,
    ReleaseCorePlusNeuralDredGate,
}

impl PreflightPolicy {
    pub fn from_flags(quick: bool, release_preflight: bool) -> Self {
        if !release_preflight {
            Self::DefaultReportOnly
        } else if quick {
            Self::ReleaseCoreSmokeNoNeuralClaim
        } else {
            Self::ReleaseCorePlusNeuralDredGate
        }
    }

    pub fn release_preflight(self) -> bool {
        !matches!(self, Self::DefaultReportOnly)
    }

    pub fn profile(self) -> &'static str {
        match self {
            Self::DefaultReportOnly => "default-report-only",
            Self::ReleaseCoreSmokeNoNeuralClaim => "release-core-smoke-no-neural-claim",
            Self::ReleaseCorePlusNeuralDredGate => "release-core-plus-neural-dred-gate",
        }
    }

    pub fn claim_note(self) -> &'static str {
        match self {
            Self::DefaultReportOnly => "default full-test report-only; no release coverage claim",
            Self::ReleaseCoreSmokeNoNeuralClaim => {
                "core smoke only; neural/DRED gates are not claimed"
            }
            Self::ReleaseCorePlusNeuralDredGate => {
                "core plus neural/DRED gate; DNN PLC and DRED format gates are claimed"
            }
        }
    }

    pub fn neural_dred_coverage_claimed(self) -> bool {
        matches!(self, Self::ReleaseCorePlusNeuralDredGate)
    }

    fn neural_asset_requirement(self) -> AssetRequirement {
        if self.neural_dred_coverage_claimed() {
            AssetRequirement::Required
        } else {
            AssetRequirement::Optional
        }
    }

    fn neural_asset_note(self) -> &'static str {
        match self {
            Self::DefaultReportOnly => {
                "report-only in default full-test; no release coverage claim"
            }
            Self::ReleaseCoreSmokeNoNeuralClaim => {
                "report-only in quick release-preflight; neural/DRED gates are not claimed"
            }
            Self::ReleaseCorePlusNeuralDredGate => {
                "required for non-quick release-preflight neural/DRED gate"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetStatus {
    PresentRequired,
    ProvisionedRequired,
    MissingRequired,
    PresentOptional,
    MissingOptional,
}

impl AssetStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            AssetStatus::PresentRequired => "present_required",
            AssetStatus::ProvisionedRequired => "provisioned_required",
            AssetStatus::MissingRequired => "missing_required",
            AssetStatus::PresentOptional => "present_optional",
            AssetStatus::MissingOptional => "missing_optional",
        }
    }

    pub fn available(self) -> bool {
        matches!(
            self,
            AssetStatus::PresentRequired
                | AssetStatus::ProvisionedRequired
                | AssetStatus::PresentOptional
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetProbe {
    pub key: &'static str,
    pub label: &'static str,
    pub requirement: AssetRequirement,
    pub status: AssetStatus,
    pub probes: Vec<String>,
    pub note: Option<String>,
    pub banner_blocking: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub policy: PreflightPolicy,
    pub release_preflight: bool,
    pub profile: &'static str,
    pub claim_note: &'static str,
    pub neural_dred_coverage_claimed: bool,
    pub assets: Vec<AssetProbe>,
}

impl Outcome {
    #[cfg(test)]
    pub fn inactive(ietf_vectors: &IetfVectorProvision) -> Self {
        capture_with_root(
            Path::new("."),
            PreflightPolicy::DefaultReportOnly,
            ietf_vectors,
        )
    }

    pub fn banner_blocking_missing(&self) -> bool {
        self.assets.iter().any(|asset| asset.banner_blocking)
    }

    #[cfg(test)]
    pub fn missing_required_for_banner(&self) -> Vec<&AssetProbe> {
        if !self.release_preflight {
            return Vec::new();
        }
        self.assets
            .iter()
            .filter(|asset| asset.banner_blocking)
            .collect()
    }
}

pub fn capture(
    workspace_root: &Path,
    policy: PreflightPolicy,
    ietf_vectors: &IetfVectorProvision,
) -> Outcome {
    capture_with_root(workspace_root, policy, ietf_vectors)
}

fn capture_with_root(
    workspace_root: &Path,
    policy: PreflightPolicy,
    ietf_vectors: &IetfVectorProvision,
) -> Outcome {
    let fixed_reference_path = workspace_root
        .join("reference")
        .join("celt")
        .join("bands.c");
    let conformance_path = workspace_root
        .join("reference")
        .join("tests")
        .join("test_opus_api.c");
    let dnn_base_paths = [
        workspace_root
            .join("reference")
            .join("dnn")
            .join("pitchdnn_data.c"),
        workspace_root
            .join("reference")
            .join("dnn")
            .join("fargan_data.c"),
        workspace_root
            .join("reference")
            .join("dnn")
            .join("plc_data.c"),
    ];
    let dred_paths = [
        workspace_root
            .join("reference")
            .join("dnn")
            .join("dred_rdovae_enc_data.c"),
        workspace_root
            .join("reference")
            .join("dnn")
            .join("dred_rdovae_dec_data.c"),
    ];

    let fixed_reference_present = fixed_reference_path.is_file();
    let dnn_base_present = dnn_base_paths.iter().all(|p| p.is_file());

    let assets = vec![
        required_file(
            "fixed_reference",
            "Fixed-point C reference",
            fixed_reference_path,
            policy.release_preflight(),
        ),
        required_file(
            "conformance_sources",
            "Upstream conformance C tests",
            conformance_path,
            policy.release_preflight(),
        ),
        ietf_asset(ietf_vectors),
        policy_files(
            "dnn_base_weights",
            "DNN base weights",
            &dnn_base_paths,
            policy,
        ),
        policy_files(
            "dred_rdovae_weights",
            "DRED RDOVAE weights",
            &dred_paths,
            policy,
        ),
        policy_derived(
            "float_deep_plc_assets",
            "Float-mode deep PLC harness assets",
            fixed_reference_present && dnn_base_present,
            "requires fixed_reference and dnn_base_weights",
            policy,
        ),
    ];

    Outcome {
        policy,
        release_preflight: policy.release_preflight(),
        profile: policy.profile(),
        claim_note: policy.claim_note(),
        neural_dred_coverage_claimed: policy.neural_dred_coverage_claimed(),
        assets,
    }
}

fn required_file(
    key: &'static str,
    label: &'static str,
    path: PathBuf,
    release_preflight: bool,
) -> AssetProbe {
    let present = path.is_file();
    let status = if present {
        AssetStatus::PresentRequired
    } else {
        AssetStatus::MissingRequired
    };
    AssetProbe {
        key,
        label,
        requirement: AssetRequirement::Required,
        status,
        probes: vec![display_path(&path)],
        note: None,
        banner_blocking: release_preflight && !present,
    }
}

fn status_for_requirement(requirement: AssetRequirement, present: bool) -> AssetStatus {
    match (requirement, present) {
        (AssetRequirement::Required, true) => AssetStatus::PresentRequired,
        (AssetRequirement::Required, false) => AssetStatus::MissingRequired,
        (AssetRequirement::Optional, true) => AssetStatus::PresentOptional,
        (AssetRequirement::Optional, false) => AssetStatus::MissingOptional,
    }
}

fn policy_files(
    key: &'static str,
    label: &'static str,
    paths: &[PathBuf],
    policy: PreflightPolicy,
) -> AssetProbe {
    let present = paths.iter().all(|p| p.is_file());
    let requirement = policy.neural_asset_requirement();
    AssetProbe {
        key,
        label,
        requirement,
        status: status_for_requirement(requirement, present),
        probes: paths.iter().map(|p| display_path(p)).collect(),
        note: Some(policy.neural_asset_note().to_string()),
        banner_blocking: requirement == AssetRequirement::Required && !present,
    }
}

fn policy_derived(
    key: &'static str,
    label: &'static str,
    present: bool,
    note: &'static str,
    policy: PreflightPolicy,
) -> AssetProbe {
    let requirement = policy.neural_asset_requirement();
    AssetProbe {
        key,
        label,
        requirement,
        status: status_for_requirement(requirement, present),
        probes: Vec::new(),
        note: Some(format!("{note}; {}", policy.neural_asset_note())),
        banner_blocking: requirement == AssetRequirement::Required && !present,
    }
}

fn ietf_asset(ietf_vectors: &IetfVectorProvision) -> AssetProbe {
    let status = match ietf_vectors.status {
        ProvisionStatus::Present => AssetStatus::PresentRequired,
        ProvisionStatus::Provisioned => AssetStatus::ProvisionedRequired,
        ProvisionStatus::Unavailable => AssetStatus::MissingRequired,
    };
    AssetProbe {
        key: "ietf_vectors",
        label: "IETF RFC 6716 / RFC 8251 vectors",
        requirement: AssetRequirement::Required,
        status,
        probes: Vec::new(),
        note: Some(
            "failure accounting remains in Stage 2 synthetic conformance result".to_string(),
        ),
        banner_blocking: false,
    }
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(root: &Path, rel: &str) {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().expect("fixture has parent")).expect("mkdir");
        fs::write(path, b"x").expect("write fixture");
    }

    fn asset<'a>(outcome: &'a Outcome, key: &str) -> &'a AssetProbe {
        outcome
            .assets
            .iter()
            .find(|asset| asset.key == key)
            .expect("asset exists")
    }

    fn write_core_assets(root: &Path) {
        write(root, "reference/celt/bands.c");
        write(root, "reference/tests/test_opus_api.c");
    }

    fn write_neural_assets(root: &Path) {
        write(root, "reference/dnn/pitchdnn_data.c");
        write(root, "reference/dnn/fargan_data.c");
        write(root, "reference/dnn/plc_data.c");
        write(root, "reference/dnn/dred_rdovae_enc_data.c");
        write(root, "reference/dnn/dred_rdovae_dec_data.c");
    }

    #[test]
    fn core_required_assets_are_present_when_probe_files_exist() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_core_assets(tmp.path());

        let outcome = capture(
            tmp.path(),
            PreflightPolicy::ReleaseCoreSmokeNoNeuralClaim,
            &IetfVectorProvision::present(),
        );

        assert_eq!(
            asset(&outcome, "fixed_reference").status,
            AssetStatus::PresentRequired
        );
        assert_eq!(
            asset(&outcome, "conformance_sources").status,
            AssetStatus::PresentRequired
        );
        assert!(!outcome.banner_blocking_missing());
    }

    #[test]
    fn release_preflight_blocks_on_missing_non_ietf_required_assets() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let outcome = capture(
            tmp.path(),
            PreflightPolicy::ReleaseCoreSmokeNoNeuralClaim,
            &IetfVectorProvision::present(),
        );

        assert_eq!(
            asset(&outcome, "fixed_reference").status,
            AssetStatus::MissingRequired
        );
        assert_eq!(
            asset(&outcome, "conformance_sources").status,
            AssetStatus::MissingRequired
        );
        assert!(outcome.banner_blocking_missing());
        let keys: Vec<&str> = outcome
            .missing_required_for_banner()
            .iter()
            .map(|asset| asset.key)
            .collect();
        assert_eq!(keys, vec!["fixed_reference", "conformance_sources"]);
    }

    #[test]
    fn missing_required_assets_are_report_only_without_release_flag() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let outcome = capture(
            tmp.path(),
            PreflightPolicy::DefaultReportOnly,
            &IetfVectorProvision::present(),
        );

        assert_eq!(
            asset(&outcome, "fixed_reference").status,
            AssetStatus::MissingRequired
        );
        assert!(!outcome.banner_blocking_missing());
        assert!(outcome.missing_required_for_banner().is_empty());
    }

    #[test]
    fn ietf_unavailable_is_missing_required_but_not_preflight_banner_blocking() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let outcome = capture(
            tmp.path(),
            PreflightPolicy::ReleaseCoreSmokeNoNeuralClaim,
            &IetfVectorProvision {
                status: ProvisionStatus::Unavailable,
                attempted_fetch: true,
                script: Some("tools/fetch_ietf_vectors.sh".into()),
                exit_code: Some(7),
                reason: Some("network unavailable".to_string()),
            },
        );

        let ietf = asset(&outcome, "ietf_vectors");
        assert_eq!(ietf.status, AssetStatus::MissingRequired);
        assert!(!ietf.banner_blocking);
        assert!(
            outcome
                .missing_required_for_banner()
                .iter()
                .all(|asset| asset.key != "ietf_vectors")
        );
    }

    #[test]
    fn default_reports_missing_neural_assets_without_banner_blocking() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_core_assets(tmp.path());
        let outcome = capture(
            tmp.path(),
            PreflightPolicy::DefaultReportOnly,
            &IetfVectorProvision::present(),
        );

        for key in [
            "dnn_base_weights",
            "dred_rdovae_weights",
            "float_deep_plc_assets",
        ] {
            let row = asset(&outcome, key);
            assert_eq!(row.status, AssetStatus::MissingOptional);
            assert!(!row.banner_blocking);
            assert_eq!(row.requirement, AssetRequirement::Optional);
        }
        assert_eq!(outcome.profile, "default-report-only");
        assert!(!outcome.release_preflight);
        assert!(!outcome.neural_dred_coverage_claimed);
        assert!(!outcome.banner_blocking_missing());
    }

    #[test]
    fn quick_release_preflight_reports_neural_assets_without_claiming_them() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_core_assets(tmp.path());

        let outcome = capture(
            tmp.path(),
            PreflightPolicy::ReleaseCoreSmokeNoNeuralClaim,
            &IetfVectorProvision::present(),
        );

        assert_eq!(outcome.profile, "release-core-smoke-no-neural-claim");
        assert!(outcome.release_preflight);
        assert!(!outcome.neural_dred_coverage_claimed);
        assert!(outcome.claim_note.contains("not claimed"));
        assert!(!outcome.banner_blocking_missing());

        for key in [
            "dnn_base_weights",
            "dred_rdovae_weights",
            "float_deep_plc_assets",
        ] {
            let row = asset(&outcome, key);
            assert_eq!(row.requirement, AssetRequirement::Optional);
            assert_eq!(row.status, AssetStatus::MissingOptional);
            assert!(!row.banner_blocking);
            assert!(row.note.as_deref().unwrap().contains("not claimed"));
        }
    }

    #[test]
    fn non_quick_release_preflight_blocks_missing_neural_assets() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_core_assets(tmp.path());

        let outcome = capture(
            tmp.path(),
            PreflightPolicy::ReleaseCorePlusNeuralDredGate,
            &IetfVectorProvision::present(),
        );

        assert_eq!(outcome.profile, "release-core-plus-neural-dred-gate");
        assert!(outcome.release_preflight);
        assert!(outcome.neural_dred_coverage_claimed);
        assert!(outcome.banner_blocking_missing());
        let keys: Vec<&str> = outcome
            .missing_required_for_banner()
            .iter()
            .map(|asset| asset.key)
            .collect();
        assert_eq!(
            keys,
            vec![
                "dnn_base_weights",
                "dred_rdovae_weights",
                "float_deep_plc_assets"
            ]
        );

        for key in [
            "dnn_base_weights",
            "dred_rdovae_weights",
            "float_deep_plc_assets",
        ] {
            let row = asset(&outcome, key);
            assert_eq!(row.requirement, AssetRequirement::Required);
            assert_eq!(row.status, AssetStatus::MissingRequired);
            assert!(row.banner_blocking);
            assert!(row.note.as_deref().unwrap().contains("required"));
        }
    }

    #[test]
    fn present_dnn_and_dred_assets_clear_non_quick_neural_blockers() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_core_assets(tmp.path());
        write_neural_assets(tmp.path());

        let outcome = capture(
            tmp.path(),
            PreflightPolicy::ReleaseCorePlusNeuralDredGate,
            &IetfVectorProvision::present(),
        );

        assert_eq!(
            asset(&outcome, "dnn_base_weights").status,
            AssetStatus::PresentRequired
        );
        assert_eq!(
            asset(&outcome, "dred_rdovae_weights").status,
            AssetStatus::PresentRequired
        );
        assert_eq!(
            asset(&outcome, "float_deep_plc_assets").status,
            AssetStatus::PresentRequired
        );
        assert!(!outcome.banner_blocking_missing());
    }

    #[test]
    fn preflight_policy_profiles_follow_release_and_quick_flags() {
        assert_eq!(
            PreflightPolicy::from_flags(false, false),
            PreflightPolicy::DefaultReportOnly
        );
        assert_eq!(
            PreflightPolicy::from_flags(true, false),
            PreflightPolicy::DefaultReportOnly
        );
        assert_eq!(
            PreflightPolicy::from_flags(true, true),
            PreflightPolicy::ReleaseCoreSmokeNoNeuralClaim
        );
        assert_eq!(
            PreflightPolicy::from_flags(false, true),
            PreflightPolicy::ReleaseCorePlusNeuralDredGate
        );
    }
}
