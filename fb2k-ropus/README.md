# fb2k-ropus — foobar2000 input component (C++)

Builds `foo_ropus.dll`: a single-DLL foobar2000 input component that
statically links the Rust `ropus-fb2k` decoder crate. This directory
holds the C++ SDK glue; the Rust decoder sources live at
`../ropus-fb2k/` and are built (as a staticlib) on demand by CMake.

M4 milestone: the component compiles, links, and registers itself as an
input for `.opus` files. It does **not** decode yet — every method logs
its name via `console::formatter` and returns an inert zero/false. M5
wires the statically-linked Rust decoder through the FFI defined in
`ropus-fb2k/include/ropus_fb2k.h`.

## Prerequisites

- **Visual Studio 2022** (recommended — this is what the HLD targets and
  what most developers will have). VS 2026 also works; use the matching
  generator string below.
- **CMake 3.20+**. The copy bundled with VS under
  `Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe` is
  fine.
- **7-Zip** on `PATH` (or installed to the default location) so the SDK
  fetch script can unpack `SDK-2025-03-07.7z`.

## One-time SDK fetch

The foobar2000 SDK is not redistributable, so it is *not* checked into the
repository. Run the fetch script from the workspace root once before you
first build:

```powershell
pwsh -File tools\fetch-fb2k-sdk.ps1
```

This downloads `SDK-2025-03-07.7z` from foobar2000.org, verifies its size,
and extracts it into `fb2k-ropus\sdk\`. The script is idempotent; pass
`-Force` to wipe and reinstall.

## Build

From the workspace root (`c:\language\mdopus`):

```powershell
cmake -S fb2k-ropus -B fb2k-ropus\build -G "Visual Studio 17 2022" -A x64
cmake --build fb2k-ropus\build --config Release
```

If you are on VS 2026, swap the generator for `"Visual Studio 18 2026"`.
Do not drop the `-A x64` — fb2k 2.0 is 64-bit and a mismatched platform
either fails to link or produces a DLL that refuses to load.

Note: a single `cmake --build` run drives `cargo build -p ropus-fb2k` as a
dependency of the C++ link step, so the Rust staticlib (`ropus_fb2k.lib`)
is (re)built as needed automatically. No separate cargo invocation required.

The output DLL lands at `fb2k-ropus\build\Release\foo_ropus.dll`.

## Install (for manual testing)

foobar2000 keeps user components under
`%APPDATA%\foobar2000-v2\user-components-x64\<name>\`. The per-component
folder does not exist until you create it; the DLL must live inside, not
alongside.

Prerequisite: foobar2000 v2 must be installed and launched at least once —
this creates `%APPDATA%\foobar2000-v2\`, which the commands below require.

For M4 rapid iteration:

```
mkdir %APPDATA%\foobar2000-v2\user-components-x64\foo_ropus
copy fb2k-ropus\build\Release\foo_ropus.dll ^
     %APPDATA%\foobar2000-v2\user-components-x64\foo_ropus\
```

Then restart foobar2000. You should see the component listed at
*Preferences → Components*; opening any `.opus` file will log
`[ropus] open …` / `[ropus] decode_run (stub → EOF)` lines to the foobar2000
console and the file will fail to play (expected for M4).

`.fb2k-component` zip packaging is an M6 deliverable; it is not done here.

## Layout

```
fb2k-ropus/
├── CMakeLists.txt              build script (drives cargo + builds pfc/SDK/component)
├── README.md                   this file
├── sdk/                        foobar2000 SDK — fetched, git-ignored
└── src/
    ├── dllmain.cpp             DECLARE_COMPONENT_VERSION + filename validator
    └── input_ropus_opus.cpp    input_singletrack_impl skeleton
```

## What the CMake build does

CMake configures three SDK libraries as static library targets, sourced
directly from the fetched SDK tree:

- `pfc` — portable foundation classes (headers only + ~30 `.cpp` files).
- `foobar2000_SDK` — service definitions and helpers.
- `foobar2000_component_client` — the `foobar2000_get_interface` entry point.

It also drives `cargo build -p ropus-fb2k` via an `add_custom_command`,
producing `target/{release,debug}/ropus_fb2k.lib` (the Rust decoder
staticlib) with an explicit `add_dependencies` edge so the MSVC linker
always sees an up-to-date lib.

These get linked into `foo_ropus.dll` alongside the prebuilt
`shared-x64.lib` (the import library for foobar2000's `shared.dll`, which
ships with fb2k itself and exposes the UTF-8 collation, logging, and crash
plumbing the SDK headers declare), plus the Rust staticlib and its
transitive Windows system deps
(`kernel32 ntdll userenv ws2_32 dbghelp` — authoritative list from
`cargo rustc -- --print native-static-libs`).

`dumpbin /dependents foo_ropus.dll` should show `shared.dll`, the MSVCRT
family, `KERNEL32.dll`, and the usual Windows imports — but **not**
`ropus_fb2k.dll`: the Rust code is compiled into `foo_ropus.dll` itself.

The CRT is pinned to `/MD` (Release) / `/MDd` (Debug) to match the SDK's
own settings — mismatched CRT flags produce a DLL that silently fails to
load in foobar2000.

## Component GUID

The authoritative `input_entry` GUID for this component is:

```
{6CA98269-9907-4C04-AE11-937D6E1933B4}
```

Do not change it. fb2k keys saved decoder-priority order by GUID; a
regenerated value would silently reset any user's input-chain ordering
on their next upgrade. The value is mirrored in
`src/input_ropus_opus.cpp::input_ropus_opus::g_get_guid`.
