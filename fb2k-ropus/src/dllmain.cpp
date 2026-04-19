// dllmain.cpp — component version declaration and filename validation.
//
// foobar2000 requires exactly one DECLARE_COMPONENT_VERSION per DLL; the
// troubleshooter uses the version string to tell different builds apart
// and the component updater uses it to skip components already installed.
// See sdk/foobar2000/SDK/componentversion.h for the macro definition.
//
// VALIDATE_COMPONENT_FILENAME registers a component_installation_validator
// so fb2k refuses to load the DLL if a user has renamed it away from the
// canonical filename (which would break the "install a newer version on
// top of an older one" flow).

#include <SDK/foobar2000.h>

// Version below is the C++ shell's own stream, independent of the Rust
// `ropus-fb2k` crate (Cargo.toml 0.2.0). Separate build artefacts on separate
// lifecycles; any alignment is an M6 packaging decision.
DECLARE_COMPONENT_VERSION(
    "ropus (Rust Opus decoder)",
    "0.0.4",
    "Development component backed by the Rust `ropus` codec.\n"
    "M4 scaffold: component loads but does not decode yet.\n"
    "Replaces foobar2000's built-in Opus decoder for .opus files "
    "once wired up (milestone M5).");

VALIDATE_COMPONENT_FILENAME("foo_input_ropus.dll");
