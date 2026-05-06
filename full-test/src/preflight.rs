//! Release preflight asset probes.
//!
//! C1 intentionally has one fixed `--release-preflight` core profile. Missing
//! fixed C reference and upstream conformance sources are banner-failing only
//! when that flag is set. IETF vector absence is shown here, but the existing
//! Stage 2 synthetic conformance failure remains the only failure accounting path
//! for vectors so totals are not double-counted.

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
    pub release_preflight: bool,
    pub profile: &'static str,
    pub assets: Vec<AssetProbe>,
}

impl Outcome {
    #[cfg(test)]
    pub fn inactive(ietf_vectors: &IetfVectorProvision) -> Self {
        capture_with_root(Path::new("."), false, ietf_vectors)
    }

    pub fn banner_blocking_missing(&self) -> bool {
        self.release_preflight && self.assets.iter().any(|asset| asset.banner_blocking)
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
    release_preflight: bool,
    ietf_vectors: &IetfVectorProvision,
) -> Outcome {
    capture_with_root(workspace_root, release_preflight, ietf_vectors)
}

fn capture_with_root(
    workspace_root: &Path,
    release_preflight: bool,
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
            release_preflight,
        ),
        required_file(
            "conformance_sources",
            "Upstream conformance C tests",
            conformance_path,
            release_preflight,
        ),
        ietf_asset(ietf_vectors),
        optional_files("dnn_base_weights", "DNN base weights", &dnn_base_paths),
        optional_files("dred_rdovae_weights", "DRED RDOVAE weights", &dred_paths),
        optional_derived(
            "float_deep_plc_assets",
            "Float-mode deep PLC harness assets",
            fixed_reference_present && dnn_base_present,
            "requires fixed_reference and dnn_base_weights",
        ),
    ];

    Outcome {
        release_preflight,
        profile: "core",
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

fn optional_files(key: &'static str, label: &'static str, paths: &[PathBuf]) -> AssetProbe {
    let present = paths.iter().all(|p| p.is_file());
    AssetProbe {
        key,
        label,
        requirement: AssetRequirement::Optional,
        status: if present {
            AssetStatus::PresentOptional
        } else {
            AssetStatus::MissingOptional
        },
        probes: paths.iter().map(|p| display_path(p)).collect(),
        note: Some("report-only in C1".to_string()),
        banner_blocking: false,
    }
}

fn optional_derived(
    key: &'static str,
    label: &'static str,
    present: bool,
    note: &'static str,
) -> AssetProbe {
    AssetProbe {
        key,
        label,
        requirement: AssetRequirement::Optional,
        status: if present {
            AssetStatus::PresentOptional
        } else {
            AssetStatus::MissingOptional
        },
        probes: Vec::new(),
        note: Some(format!("{note}; report-only in C1")),
        banner_blocking: false,
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

    #[test]
    fn core_required_assets_are_present_when_probe_files_exist() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write(tmp.path(), "reference/celt/bands.c");
        write(tmp.path(), "reference/tests/test_opus_api.c");

        let outcome = capture(tmp.path(), true, &IetfVectorProvision::present());

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
        let outcome = capture(tmp.path(), true, &IetfVectorProvision::present());

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
        let outcome = capture(tmp.path(), false, &IetfVectorProvision::present());

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
            true,
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
    fn optional_dnn_and_dred_assets_never_banner_block() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let outcome = capture(tmp.path(), true, &IetfVectorProvision::present());

        for key in [
            "dnn_base_weights",
            "dred_rdovae_weights",
            "float_deep_plc_assets",
        ] {
            let row = asset(&outcome, key);
            assert_eq!(row.status, AssetStatus::MissingOptional);
            assert!(!row.banner_blocking);
        }
    }

    #[test]
    fn optional_assets_become_present_when_all_probe_files_exist() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write(tmp.path(), "reference/celt/bands.c");
        write(tmp.path(), "reference/dnn/pitchdnn_data.c");
        write(tmp.path(), "reference/dnn/fargan_data.c");
        write(tmp.path(), "reference/dnn/plc_data.c");
        write(tmp.path(), "reference/dnn/dred_rdovae_enc_data.c");
        write(tmp.path(), "reference/dnn/dred_rdovae_dec_data.c");

        let outcome = capture(tmp.path(), false, &IetfVectorProvision::present());

        assert_eq!(
            asset(&outcome, "dnn_base_weights").status,
            AssetStatus::PresentOptional
        );
        assert_eq!(
            asset(&outcome, "dred_rdovae_weights").status,
            AssetStatus::PresentOptional
        );
        assert_eq!(
            asset(&outcome, "float_deep_plc_assets").status,
            AssetStatus::PresentOptional
        );
    }
}
