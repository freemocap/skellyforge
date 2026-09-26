# Native Ceres infrastructure

This is a compiled C++ extension, not PyCeres. It currently exposes only an
automatic-differentiation scalar smoke solve. No skeleton fitting changes.

## Development

Run from the SkellyForge repository. Bootstrap build tools before building the
package itself:

```powershell
uv sync --group native --extra recording-viewer --no-install-project
uv sync --group native --extra recording-viewer --no-editable
uv run --no-sync poe native-check
uv run --no-sync poe native-install
uv run --no-sync poe test-native
uv run --no-sync poe test-native-python
uv run --no-sync poe test
uv run --no-sync poe build-wheel
```

Windows requires Visual Studio 2022 Build Tools with Desktop development with
C++ and a Windows SDK. CMake discovers Visual Studio from ordinary PowerShell.
Linux/macOS require a C++17 toolchain. `native-check` checks prerequisites;
configuration is the definitive compiler/linker check.

Use `--no-editable` when syncing the project: scikit-build-core's redirect-mode
editable import hook bypasses Forge's beartype import instrumentation. Source
edits are still used when running Python from this checkout; `native-install`
updates its compiled module. Use `uv run --no-sync` for tasks so uv does not
replace the regular installation with a redirect-mode editable installation.

`native-build` configures and incrementally compiles. `native-install` also copies
the generated extension into the checkout's Python package, because that package
is imported when running tests here. It does not install globally. Python must
be restarted to load a rebuilt extension. Generated modules and notices are
ignored. Wheel builds install into wheel staging instead; no manual copy is
needed for users installing a wheel.

`build/native/` contains local CMake products and dependency sources.
`build/wheel/` contains packaging builds. `dist/` contains wheels. No Git
submodules, clones, or update commands run as part of these tasks. Release
archives are fetched by CMake and verified against SHA256 hashes.

## Layout and ownership

- `include/skellyforge/`: native declarations.
- `src/`: implementation, including Ceres AutoDiff residuals.
- `bindings/`: pybind11 boundary; exposes private `skellyforge._native`.
- `tests/`: native executable smoke checks, run through CTest.
- `cmake/dependencies.cmake`: pinned dependency archives and features.
- Python boundary checks remain in `skellyforge/tests/test_native_extension.py`.

Ceres 2.2.0 and Eigen 3.4.0 are pinned for this initial build. Ceres is static,
uses its bundled minimal logging implementation and Eigen sparse support, and
does not require CUDA, SuiteSparse, LAPACK, or a separate glog installation.
This is a reproducible initial build configuration, not a performance conclusion
or a promise never to upgrade. Dependency notices accompany the module.

The smoke solve exercises compilation, linking, Ceres autodifferentiation, solve
termination, returned results and C++ exception translation. Additional native Python tests now exercise quaternion rigid fitting, exact point
linkages and temporal residual blocks for rigid and linked sequences. Human skeleton
FK, non-rigid connections and joint bounds are not yet covered by these experiments.

## Packaging

scikit-build-core replaces flit_core, calls CMake, and packages the existing
Python tree plus the extension. Version metadata still comes from __init__.py.
Poe is the task entry point; CMake owns native dependency/build relationships.
`build-wheel` uses the installed native tools (`--no-build-isolation`). Normal
isolated wheel builds declare their own build requirements in pyproject.toml.

The native CI workflow builds and tests without publishing. Portable repaired
Linux/macOS wheel distribution and the existing release-publishing workflow need
a subsequent integration review before publishing this native package.


## Release performance

The native lab must use an optimized Release build. An existing local cache was
found with empty C/C++ Release flag strings, which disabled the compiler defaults
even though the configuration was named Release. The top-level CMake file now
restores compiler defaults for that specific empty-cache state before `project()`;
nonempty custom flags remain untouched. On MSVC, verify `/O2 /Ob2 /DNDEBUG` in
CMakeCache.txt and `/O2` plus `NDEBUG` in the actual compiler command log. Use
`poe native-install` to rebuild both Ceres and Forge after this repair. Do not
judge solver performance from the previous unoptimized binary.
