cmake_minimum_required(VERSION 3.20)
project(fb2k_ropus CXX)

# --------------------------------------------------------------------------
# foo_ropus — single-DLL foobar2000 input component
# --------------------------------------------------------------------------
#
# This CMake script builds foo_ropus.dll: a 64-bit foobar2000 input component
# for .opus files. M4b collapses the old two-DLL split: Rust is now a
# staticlib linked into this component directly. M5 fills in the stub decode
# path against those already-linked symbols.
#
# The foobar2000 SDK ships as MSBuild .vcxproj files rather than CMake.
# Instead of shelling out to MSBuild we rebuild the three static helper
# libraries we need (pfc, foobar2000_SDK, foobar2000_component_client) as
# CMake targets that glob each SDK subdir's .cpp files, and link the
# prebuilt shared-x64.lib for the parts that live in foobar2000's
# shared.dll at runtime (logging hooks, UTF-8 compare, crash handler).
#
# The Rust codec (workspace crate ropus-fb2k) is built as a staticlib by a
# custom command that invokes `cargo build`; the resulting ropus_fb2k.lib
# is linked directly into foo_ropus.dll. No runtime DLL load — single
# artifact to ship.
#
# The SDK itself is fetched via tools/fetch-fb2k-sdk.ps1 — see README.md.

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "foo_ropus is 64-bit only. Pass -A x64 when configuring.")
endif()

if(NOT WIN32)
    message(FATAL_ERROR "foo_ropus is a foobar2000 component; Windows only.")
endif()

# foobar2000's DLLs are built with the dynamic CRT (/MD). Plugins that link
# a mismatched CRT silently fail to load. Pin it globally. Rust's MSVC
# toolchain defaults to /MD when +crt-static is not set; the C++ /MD here
# drags msvcrt.lib into the final link, which is what satisfies CRT symbols
# referenced by the Rust objects. See the `target_link_libraries` block
# below for the full story.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# CRT mix + optimization mix caveat: Rust doesn't emit a /DEFAULTLIB
# directive for the CRT in its staticlib bytes. The C++ side's /MD
# (release) or /MDd (debug) drags in whichever CRT via its own objects.
# In Debug, that's msvcrtd — but a cargo `dev` staticlib compiled without
# matching flags risks cross-allocator heap corruption and duplicate
# errno/iob. MinSizeRel and RelWithDebInfo compile the C++ side at /O1
# or /O2 /Zi while Rust still produces a release staticlib, so the
# optimization levels are incoherent even where the CRT happens to line
# up. M4b validated Release only. Note this is M5-forward protection:
# Debug foo_ropus.dll currently has zero Rust symbols (M4b wires the
# staticlib in but doesn't call into it), so the mismatch is theoretical
# until M5 starts invoking the Rust decoder. Surface a warning on any
# non-Release config so developers know what they've picked.
set(_ROPUS_UNTESTED_CONFIGS "Debug" "MinSizeRel" "RelWithDebInfo")
if(CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE IN_LIST _ROPUS_UNTESTED_CONFIGS)
    message(WARNING
        "Non-Release config '${CMAKE_BUILD_TYPE}' mixes a Rust release "
        "staticlib with C++ objects built at a different optimization and/or "
        "CRT level. In M4b this is silent — the DLL contains no Rust symbols "
        "yet — but in M5 forward it becomes a live runtime hazard (CRT "
        "mismatch, incoherent optimization). Release is the supported config. "
        "See wrk_journals/2026.04.19 - JRN - ropus-fb2k-implementation.md §M4b.")
elseif(CMAKE_CONFIGURATION_TYPES)
    set(_ROPUS_UNTESTED_IN_USE "")
    foreach(_cfg IN LISTS _ROPUS_UNTESTED_CONFIGS)
        if(_cfg IN_LIST CMAKE_CONFIGURATION_TYPES)
            list(APPEND _ROPUS_UNTESTED_IN_USE "${_cfg}")
        endif()
    endforeach()
    if(_ROPUS_UNTESTED_IN_USE)
        message(STATUS
            "foo_ropus: non-Release configs listed but untested — "
            "${_ROPUS_UNTESTED_IN_USE}. Build --config Release for the "
            "supported path (M4b ships no Rust symbols in Debug/MinSizeRel/"
            "RelWithDebInfo; M5 forward will make the CRT/optimization "
            "mismatch live).")
    endif()
    unset(_ROPUS_UNTESTED_IN_USE)
endif()
unset(_ROPUS_UNTESTED_CONFIGS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Confirm the SDK tree is where we expect it. Fetched by
# tools/fetch-fb2k-sdk.ps1 — a fresh clone has only sdk/ as a .gitignored
# directory. If it's missing, fail with a pointer to the fetch script.
set(SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/sdk")
if(NOT EXISTS "${SDK_ROOT}/foobar2000/SDK/foobar2000.h")
    message(FATAL_ERROR
        "foobar2000 SDK not found at ${SDK_ROOT}. "
        "Run tools/fetch-fb2k-sdk.ps1 from the workspace root first.")
endif()

set(FB2K_ROOT  "${SDK_ROOT}/foobar2000")
set(PFC_ROOT   "${SDK_ROOT}/pfc")
set(SDK_INC    "${FB2K_ROOT}/SDK")
set(CC_ROOT    "${FB2K_ROOT}/foobar2000_component_client")

# Compile definitions matching the SDK's .vcxproj settings for Release|x64.
# UNICODE + _UNICODE switch Win32 APIs to wide-char. The SDK deliberately
# pulls a full <windows.h> (it uses timeGetTime / mmsystem.h and expects
# no NOMINMAX / WIN32_LEAN_AND_MEAN trimming), so we mirror that — adding
# either macro breaks the pfc build.
set(FB2K_COMMON_DEFS
    UNICODE
    _UNICODE
    _CRT_SECURE_NO_WARNINGS
)

# --------------------------------------------------------------------------
# pfc — portable foundation classes
# --------------------------------------------------------------------------
#
# Glob source matches the .vcxproj layout. Excluded: stdafx.cpp (PCH stub
# we don't use), nix-objects.cpp + synchro_nix.cpp + obj-c.mm (Unix/macOS
# only — the .vcxproj marks them ExcludedFromBuild on Windows).
#
# Intentionally INCLUDED: pfc-fb2k-hooks.cpp. The .vcxproj excludes it only
# from the "Release FB2K" configuration, which is what foobar2000 itself
# uses to link against the hooks exported from shared.dll. Plugins build
# plain "Release", where pfc provides its own hooks that call the local
# crashImpl / winFormatSystemErrorMessageImpl in pfc/other.cpp.

file(GLOB PFC_SOURCES CONFIGURE_DEPENDS "${PFC_ROOT}/*.cpp")
list(FILTER PFC_SOURCES EXCLUDE REGEX "(stdafx|nix-objects|synchro_nix)\\.cpp$")

add_library(pfc STATIC ${PFC_SOURCES})
target_include_directories(pfc PUBLIC "${SDK_ROOT}")
target_compile_definitions(pfc PUBLIC ${FB2K_COMMON_DEFS})

# --------------------------------------------------------------------------
# foobar2000_SDK — the service definitions + helpers (StaticLibrary)
# --------------------------------------------------------------------------
#
# Same glob pattern as pfc. Excluded: stdafx.cpp (PCH stub we don't use).
# The SDK directory has no Unix-only files to filter here.

file(GLOB SDK_SOURCES CONFIGURE_DEPENDS "${SDK_INC}/*.cpp")
list(FILTER SDK_SOURCES EXCLUDE REGEX "stdafx\\.cpp$")

add_library(foobar2000_SDK STATIC ${SDK_SOURCES})
target_include_directories(foobar2000_SDK PUBLIC "${FB2K_ROOT}" "${SDK_ROOT}")
target_compile_definitions(foobar2000_SDK PUBLIC ${FB2K_COMMON_DEFS})
target_link_libraries(foobar2000_SDK PUBLIC pfc)

# --------------------------------------------------------------------------
# foobar2000_component_client — the fb2k ABI entry point
# --------------------------------------------------------------------------
#
# Contains the __declspec(dllexport) foobar2000_get_interface that fb2k
# looks up via GetProcAddress when it loads the component DLL. The single
# translation unit also carries the core_api:: shim that marshals through
# the g_foobar2000_api pointer fb2k passes in.

add_library(foobar2000_component_client STATIC "${CC_ROOT}/component_client.cpp")
target_include_directories(foobar2000_component_client PUBLIC "${FB2K_ROOT}" "${SDK_ROOT}")
target_compile_definitions(foobar2000_component_client PUBLIC ${FB2K_COMMON_DEFS})
target_link_libraries(foobar2000_component_client PUBLIC foobar2000_SDK pfc)

# --------------------------------------------------------------------------
# ropus_fb2k — Rust decoder staticlib
# --------------------------------------------------------------------------
#
# Drives `cargo build` on the workspace `ropus-fb2k` crate and links the
# resulting staticlib into foo_ropus.dll. Cargo is invoked with --release
# when the selected config isn't Debug; plain `cargo build` otherwise.
# Output lands in the workspace's target/{release,debug}/ropus_fb2k.lib.
#
# We use an always-run `add_custom_target(... ALL ...)` plus
# `add_dependencies` rather than `add_custom_command(OUTPUT ...)` (which
# short-circuits on the output mtime and misses .rs edits entirely) or the
# `corrosion` CMake package (one fewer external dep on a one-off Windows
# component). The ALL-target invokes cargo on every build regardless of
# MSBuild's up-to-date check; cargo's own cache is the authoritative
# staleness oracle. Measured overhead: ~5s for a direct no-op
# `cargo build --release -p ropus-fb2k` (file-lock wait on the workspace
# target dir dominates), rising to ~10–12s through the `cmake --build`
# wrapper per cycle; occasionally the custom-target's invocation args
# differ enough from a prior direct cargo call to flip the profile hash
# and trigger a full recompile despite no source change. That's tolerable
# for dev iteration — the alternative is listing every `.rs` under
# DEPENDS and accepting that missing one silently produces stale `.lib`s.

set(WORKSPACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Per-config cargo profile. Release is the default for Visual Studio
# generator when CMAKE_BUILD_TYPE isn't set (multi-config picks it at build
# time), so branch on the generator expression rather than CMAKE_BUILD_TYPE
# at configure time.
#
# Under plain `VERBATIM`, an empty genex expansion survives as a literal
# `""` argument that cargo rejects (`unexpected argument '' found`).
# Quoting the genex and adding `COMMAND_EXPAND_LISTS` changes the
# interpretation: each COMMAND argument is now parsed as a
# semicolon-separated list, and an empty-string element is a zero-element
# list which CMake drops before the shell command is emitted. In Release
# the genex yields `--release` (one-element list → one arg); in Debug it
# yields an empty string (zero-element list → no arg emitted). Other
# shapes exist (a conditional `set()` picking a variable, for example) —
# this one's just the cleanest.
set(ROPUS_TARGET_SUBDIR "$<IF:$<CONFIG:Debug>,debug,release>")
set(ROPUS_STATIC_LIB    "${WORKSPACE_ROOT}/target/${ROPUS_TARGET_SUBDIR}/ropus_fb2k.lib")

add_custom_target(ropus_fb2k_staticlib ALL
    # Pre-remove cdylib artefacts left behind when the crate-type flipped
    # from cdylib to staticlib in M4b — cargo does not clean them on its
    # own across crate-type changes. Harmless on clean machines; proactive
    # on dev machines that built Debug under the old shape.
    COMMAND ${CMAKE_COMMAND} -E rm -f
        "${WORKSPACE_ROOT}/target/${ROPUS_TARGET_SUBDIR}/ropus_fb2k.dll"
        "${WORKSPACE_ROOT}/target/${ROPUS_TARGET_SUBDIR}/ropus_fb2k.dll.lib"
        "${WORKSPACE_ROOT}/target/${ROPUS_TARGET_SUBDIR}/ropus_fb2k.pdb"
    COMMAND cargo build "$<$<NOT:$<CONFIG:Debug>>:--release>" -p ropus-fb2k
    WORKING_DIRECTORY "${WORKSPACE_ROOT}"
    COMMENT "Building ropus-fb2k staticlib ($<IF:$<CONFIG:Debug>,dev,release>)"
    VERBATIM
    COMMAND_EXPAND_LISTS
    USES_TERMINAL
)

# --------------------------------------------------------------------------
# foo_ropus.dll — the component
# --------------------------------------------------------------------------

add_library(foo_ropus SHARED
    src/dllmain.cpp
    src/input_ropus_opus.cpp
)

add_dependencies(foo_ropus ropus_fb2k_staticlib)

# fb2k ignores CMake's default DLL name (foo_ropus.dll would already
# be right) but an explicit OUTPUT_NAME guards against anyone sneaking in
# a prefix/suffix change at the project level.
set_target_properties(foo_ropus PROPERTIES
    OUTPUT_NAME "foo_ropus"
    PREFIX ""
    SUFFIX ".dll"
)

target_compile_definitions(foo_ropus PRIVATE _WINDLL)

# Rust staticlib's C ABI header. The `..` path is stable because
# fb2k-ropus/ and ropus-fb2k/ are sibling directories under the workspace
# root — do not collapse into a single name without a second pair of eyes.
target_include_directories(foo_ropus PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../ropus-fb2k/include"
)

# Per-target split: our code compiles clean at /W4 /permissive- (the
# `(void)p_xxx;` casts in input_ropus_opus.cpp deliberately silence C4100
# on stub parameters). The three SDK static libs above stay at CMake's
# default warning level (/W3) — we don't own those sources and have no
# business chasing their warnings.
#
# SDK headers still instantiate template stubs inside *our* TU when we
# register via input_singletrack_factory_t<>, so SDK-owned warnings can
# surface on our build log even though the offending source lines belong
# to sdk/foobar2000/SDK/input_impl.h. The SDK is reached through
# angle-bracket includes (see foobar2000.h), so /external:anglebrackets
# + /external:W0 marks it as external and silences parse-time warnings
# (C4100) from there. /wd4702 additionally suppresses the codegen-phase
# "unreachable code" warning on input_impl.h's intentional `throw;` +
# trailing return patterns — codegen warnings escape /external:W0.
if(MSVC)
    target_compile_options(foo_ropus PRIVATE
        /W4 /permissive-
        /external:anglebrackets
        /external:W0
        /wd4702
    )
endif()

# Windows system libs required by the Rust staticlib. Authoritative list
# obtained from `cargo rustc --release -p ropus-fb2k -- --print native-static-libs`
# on 2026-04-19 (Rust 1.x on x86_64-pc-windows-msvc). The command prints:
#     kernel32.lib ntdll.lib userenv.lib ws2_32.lib dbghelp.lib /defaultlib:msvcrt
# The trailing /defaultlib:msvcrt is a hint from cargo, not a directive
# embedded in the .lib file itself. We do not need to pass it explicitly —
# the C++ side's /MD compile already pulls msvcrt.lib into the link, which
# satisfies the CRT for both the C++ TUs and the Rust objects that reference
# CRT symbols (malloc, printf, _iob). winmm stays in the list because pfc
# (C++ side) uses timeGetTime/timeBeginPeriod independently of Rust. If Rust
# deps change, re-run the command above and update this list.
target_link_libraries(foo_ropus PRIVATE
    foobar2000_component_client
    foobar2000_SDK
    pfc
    "${FB2K_ROOT}/shared/shared-x64.lib"
    winmm   # timeGetTime / timeBeginPeriod used by pfc::media_timer
    # Rust staticlib + its transitive Windows system deps
    "${ROPUS_STATIC_LIB}"
    kernel32
    ntdll
    userenv
    ws2_32
    dbghelp
)

# shared.dll is part of any foobar2000 install; we resolve against
# shared-x64.lib at link time and fb2k provides the runtime DLL. Record
# the import so `dumpbin /dependents` is informative.

# Put the final DLL directly into the build directory's per-config folder
# (Release/, Debug/) so Arthur can find it without chasing CMake's default
# generator expressions.
set_target_properties(foo_ropus PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY            "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG      "${CMAKE_BINARY_DIR}/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE    "${CMAKE_BINARY_DIR}/Release"
)
