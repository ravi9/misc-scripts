#!/usr/bin/env bash
#
# test-ov-release-ci.sh
#
# Local validation for the ubuntu-24-openvino release packaging in
# llama.cpp/.github/workflows/release.yml. Mirrors the CI job:
#   1. Configure llama.cpp with -DGGML_OPENVINO=ON and RPATH=$ORIGIN using the
#      Ninja generator (same flags as CI, incl. $CMAKE_ARGS).
#   2. Build into a SEPARATE build dir (build/ReleaseOV-CI) so an existing
#      build/ReleaseOV is left intact.
#   3. Run the EXACT pack-step from release.yml (bundle OpenVINO + TBB .so
#      files, plugin xml/cache.json, libOpenCL.so.1, plus LICENSE) and tar it.
#   4. Extract the tarball to a clean folder and run ./llama-cli -h with a
#      sanitised environment (no setupvars.sh, unset LD_LIBRARY_PATH) to prove
#      the package is self-contained. Also runs ldd to flag unresolved deps.
#
# Pass: exit code 0 with help text printed -> package is self-contained.
# Fail: non-zero exit (e.g. "error while loading shared libraries") or any
#       'not found' in ldd -> a .so dependency is still missing.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR/llama.cpp}"
OV_ROOT="${OV_ROOT:-/opt/intel/openvino_2026.2.1}"
BUILD_DIR="${BUILD_DIR:-build/ReleaseOV-CI}"
TAR_PATH="${TAR_PATH:-$SCRIPT_DIR/test-llama-bin-ubuntu-openvino-x64.tar.gz}"
TEST_DIR="${TEST_DIR:-$SCRIPT_DIR/test-openvino-extracted}"
OPENCL_LOADER="${OPENCL_LOADER:-/usr/lib/x86_64-linux-gnu/libOpenCL.so.1}"
SKIP_BUILD="${SKIP_BUILD:-0}"   # set to 1 to rerun only Pack + Test

section() {
    echo ""
    echo "============================================================"
    echo " $1"
    echo "============================================================"
}

fail() { echo "ERROR: $1" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 0. Sanity checks
# -----------------------------------------------------------------------------
section "Checking prerequisites"

[ -f "$REPO_ROOT/CMakeLists.txt" ] || fail "llama.cpp source not found at $REPO_ROOT"
[ -f "$OV_ROOT/setupvars.sh" ]     || fail "OpenVINO toolkit not found at $OV_ROOT"
command -v cmake >/dev/null        || fail "cmake not found"
command -v ninja >/dev/null        || fail "ninja not found"

echo "  Repo      : $REPO_ROOT"
echo "  OpenVINO  : $OV_ROOT"
echo "  Build dir : $REPO_ROOT/$BUILD_DIR  (separate from build/ReleaseOV)"
echo "  Tarball   : $TAR_PATH"
echo "  Test dir  : $TEST_DIR"

cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# 1-3. Configure + build - same flags as the release.yml CI step
# -----------------------------------------------------------------------------
if [ "$SKIP_BUILD" != "1" ]; then
    section "Importing OpenVINO setupvars into env"
    set +u
    # shellcheck disable=SC1091
    source "$OV_ROOT/setupvars.sh"
    set -u

    section "CMake configure (CI-equivalent flags)"
    rm -rf "$BUILD_DIR"
    # $ORIGIN is single-quoted so the shell does not expand it; CMake bakes the
    # literal RPATH into the binaries so they load the bundled sibling .so files.
    cmake -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_OPENVINO=ON \
        -DCMAKE_INSTALL_RPATH='$ORIGIN' \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DGGML_RPC=ON \
        || fail "CMake configure failed"

    section "Building (this can take a while on a cold cache)"
    cmake --build "$BUILD_DIR" --config Release --parallel "$(nproc)" \
        || fail "CMake build failed"
fi

# -----------------------------------------------------------------------------
# 4. Pack - same logic as release.yml ubuntu-24-openvino Pack step
# -----------------------------------------------------------------------------
section "Pack artifacts (mirrors release.yml ubuntu-24-openvino Pack step)"

dest="./$BUILD_DIR/bin"
[ -x "$dest/llama-cli" ] || fail "llama-cli not found at $dest - did the build succeed?"

ov_lib="$OV_ROOT/runtime/lib/intel64"

# Bundle OpenVINO runtime libs + TBB. Binaries built with RPATH=$ORIGIN
# load these siblings without setupvars.sh / LD_LIBRARY_PATH.
# Exclude frontends (onnx, paddle, pytorch, tensorflow*) - not needed for GGUF
# models, saves ~16MB. NPU compiler is included for self-contained NPU inference.
cp -P "$ov_lib"/libopenvino.so* \
      "$ov_lib"/libopenvino_c.so* \
      "$ov_lib"/libopenvino_*_plugin.so \
      "$ov_lib"/libopenvino_intel_npu_compiler*.so \
      "$OV_ROOT/runtime/3rdparty/tbb/lib"/*.so* \
      "$dest"/

# OpenCL ICD loader (mirrors the OpenCL.dll bundled in the Windows job).
cp -P "$OPENCL_LOADER"* "$dest" 2>/dev/null || true
cp "$ov_lib/cache.json" "$dest" 2>/dev/null || true

# OpenVINO licensing
cp -r "$OV_ROOT/docs/licensing" "$dest/openvino-licensing"

cp LICENSE "$dest"

rm -f "$TAR_PATH"
tar -czf "$TAR_PATH" --transform "s,^\.,llama-test," -C "$dest" .
echo "  Tarball created: $TAR_PATH ($(du -h "$TAR_PATH" | cut -f1))"

# -----------------------------------------------------------------------------
# 5. Extract tarball to a clean test folder
# -----------------------------------------------------------------------------
section "Extracting tarball to clean test folder"

rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"
tar -xzf "$TAR_PATH" -C "$TEST_DIR"

test_dir_inner="$TEST_DIR/llama-test"
test_exe="$test_dir_inner/llama-cli"
[ -x "$test_exe" ] || fail "llama-cli not found at $test_exe after extract"
echo "  Extracted to: $test_dir_inner"
echo "  Files: $(find "$test_dir_inner" -maxdepth 1 -type f | wc -l)"

# -----------------------------------------------------------------------------
# 6. Run ./llama-cli -h with a SANITISED env (no setupvars, unset LD_LIBRARY_PATH)
#    This proves the package works on a machine with no OpenVINO install.
# -----------------------------------------------------------------------------
section "Running ./llama-cli -h with sanitised env (LD_LIBRARY_PATH unset)"

echo ""
echo "--- ldd ./llama-cli (unresolved deps appear as 'not found') ---"
( cd "$test_dir_inner" && env -u LD_LIBRARY_PATH ldd ./llama-cli ) || true
missing="$( cd "$test_dir_inner" && env -u LD_LIBRARY_PATH ldd ./llama-cli 2>/dev/null | grep -c 'not found' )"

echo ""
echo "--- ./llama-cli -h (first 30 lines) ---"
stdout_file="$(mktemp)"
( cd "$test_dir_inner" && env -u LD_LIBRARY_PATH ./llama-cli -h ) >"$stdout_file" 2>&1
exit_code=$?
head -n 30 "$stdout_file"

echo ""
echo "  Exit code   : $exit_code"
echo "  'not found' : $missing"

# -----------------------------------------------------------------------------
# 7. Verdict
# -----------------------------------------------------------------------------
section "Verdict"

if [ "$exit_code" -eq 0 ] && [ "$missing" -eq 0 ] && [ -s "$stdout_file" ]; then
    echo "  PASS - package is self-contained (no OpenVINO install required)"
    rm -f "$stdout_file"
    exit 0
else
    echo "  FAIL - package is not self-contained"
    if [ "$missing" -ne 0 ]; then
        echo "  $missing shared-library dependency(ies) still unresolved."
        echo "  Inspect with:"
        echo "    ( cd \"$test_dir_inner\" && env -u LD_LIBRARY_PATH ldd ./llama-cli )"
        echo "    ( cd \"$test_dir_inner\" && env -u LD_LIBRARY_PATH ldd ./libggml-openvino.so )"
    fi
    echo "--- full output ---"
    cat "$stdout_file"
    rm -f "$stdout_file"
    exit 1
fi
