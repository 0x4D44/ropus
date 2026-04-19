cmake_minimum_required(VERSION 3.20)
project(fb2k_ropus CXX)

# --------------------------------------------------------------------------
# foo_input_ropus — M4 scaffold
# --------------------------------------------------------------------------
#
# This CMake script builds foo_input_ropus.dll: the M4 milestone is a
# skeleton foobar2000 input component that registers but does nothing (each
# method logs its name via console::formatter). The Rust-side decoder
# (ropus_fb2k.dll) is NOT wired yet — that lands in M5.
#
# The foobar2000 SDK ships as MSBuild .vcxproj files rather than CMake.
# Instead of shelling out to MSBuild we rebuild the three static helper
# libraries we need (pfc, foobar2000_SDK, foobar2000_component_client) as
# CMake targets that glob each SDK subdir's .cpp files, and link the
# prebuilt shared-x64.lib for the parts that live in foobar2000's
# shared.dll at runtime (logging hooks, UTF-8 compare, crash handler).
#
# The SDK itself is fetched via tools/fetch-fb2k-sdk.ps1 — see README.md.

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "foo_input_ropus is 64-bit only. Pass -A x64 when configuring.")
endif()

if(NOT WIN32)
    message(FATAL_ERROR "foo_input_ropus is a foobar2000 component; Windows only.")
endif()

# foobar2000's DLLs are built with the dynamic CRT (/MD). Plugins that link
# a mismatched CRT silently fail to load. Pin it globally.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

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
# foo_input_ropus.dll — the component
# --------------------------------------------------------------------------

add_library(foo_input_ropus SHARED
    src/dllmain.cpp
    src/input_ropus_opus.cpp
)

# fb2k ignores CMake's default DLL name (foo_input_ropus.dll would already
# be right) but an explicit OUTPUT_NAME guards against anyone sneaking in
# a prefix/suffix change at the project level.
set_target_properties(foo_input_ropus PROPERTIES
    OUTPUT_NAME "foo_input_ropus"
    PREFIX ""
    SUFFIX ".dll"
)

target_compile_definitions(foo_input_ropus PRIVATE _WINDLL)

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
    target_compile_options(foo_input_ropus PRIVATE
        /W4 /permissive-
        /external:anglebrackets
        /external:W0
        /wd4702
    )
endif()

target_link_libraries(foo_input_ropus PRIVATE
    foobar2000_component_client
    foobar2000_SDK
    pfc
    "${FB2K_ROOT}/shared/shared-x64.lib"
    winmm   # timeGetTime / timeBeginPeriod used by pfc::media_timer
)

# shared.dll is part of any foobar2000 install; we resolve against
# shared-x64.lib at link time and fb2k provides the runtime DLL. Record
# the import so `dumpbin /dependents` is informative.

# Put the final DLL directly into the build directory's per-config folder
# (Release/, Debug/) so Arthur can find it without chasing CMake's default
# generator expressions.
set_target_properties(foo_input_ropus PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY            "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG      "${CMAKE_BINARY_DIR}/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE    "${CMAKE_BINARY_DIR}/Release"
)
