use std::collections::BTreeSet;
use std::path::Path;

/// Pure manifest for C-reference compilation and Cargo invalidation.
///
/// Every compiled source is also a file-level watch. Directory watches cover
/// headers and source files that may be added to the pinned reference tree.
#[derive(Debug)]
pub(crate) struct ReferenceBuildManifest {
    sources: Vec<&'static str>,
    watch_directories: Vec<&'static str>,
}

impl ReferenceBuildManifest {
    pub(crate) fn try_new(
        sources: impl IntoIterator<Item = &'static str>,
        watch_directories: impl IntoIterator<Item = &'static str>,
    ) -> Result<Self, String> {
        let sources: Vec<_> = sources.into_iter().collect();
        let watch_directories: Vec<_> = watch_directories.into_iter().collect();
        validate_relative_unique("source", &sources)?;
        validate_relative_unique("watch directory", &watch_directories)?;
        Ok(Self {
            sources,
            watch_directories,
        })
    }

    pub(crate) fn sources(&self) -> &[&'static str] {
        &self.sources
    }

    pub(crate) fn watched_sources(&self) -> &[&'static str] {
        &self.sources
    }

    pub(crate) fn watch_directories(&self) -> &[&'static str] {
        &self.watch_directories
    }
}

fn validate_relative_unique(kind: &str, paths: &[&str]) -> Result<(), String> {
    let mut unique = BTreeSet::new();
    for path in paths {
        if path.is_empty()
            || Path::new(path).is_absolute()
            || path.split('/').any(|part| part == "..")
        {
            return Err(format!("invalid {kind} path: {path:?}"));
        }
        if !unique.insert(*path) {
            return Err(format!("duplicate {kind} path: {path}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ReferenceBuildManifest;

    #[test]
    fn compiled_sources_have_exactly_one_file_watch() {
        let manifest = ReferenceBuildManifest::try_new(
            ["celt/bands.c", "src/opus_encoder.c"],
            ["include", "celt", "src"],
        )
        .unwrap();

        assert_eq!(manifest.sources(), manifest.watched_sources());
        assert_eq!(manifest.watch_directories(), &["include", "celt", "src"]);
    }

    #[test]
    fn duplicate_source_is_rejected_before_compilation() {
        let error =
            ReferenceBuildManifest::try_new(["src/opus_encoder.c", "src/opus_encoder.c"], ["src"])
                .unwrap_err();
        assert!(error.contains("duplicate source"));
    }

    #[test]
    fn escaping_reference_root_is_rejected() {
        let error = ReferenceBuildManifest::try_new(["../outside.c"], ["src"]).unwrap_err();
        assert!(error.contains("invalid source"));
    }
}
