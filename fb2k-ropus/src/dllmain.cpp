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

// Version below is the single user-facing component version shown in fb2k's
// Components dialog. The Rust ropus-fb2k crate has its own internal version
// (Cargo.toml 0.3.0); post-M4b it's statically linked into this DLL and no
// longer surfaces as a separate artefact. Bump this string when the
// component's user-visible behaviour changes.
DECLARE_COMPONENT_VERSION(
    "ropus (Rust Opus decoder)",
    "0.1.0",
    "Development component backed by the Rust `ropus` codec.\n"
    "M4 scaffold: component loads but does not decode yet.\n"
    "Replaces foobar2000's built-in Opus decoder for .opus files "
    "once wired up (milestone M5).");

VALIDATE_COMPONENT_FILENAME("foo_ropus.dll");
