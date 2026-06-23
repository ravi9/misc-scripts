# Test Dockerfile for self-contained OpenVINO bundle
# Verifies that the optimized bundle works without system OpenVINO installed

ARG http_proxy=
ARG https_proxy=

FROM ubuntu:24.04

ARG http_proxy
ARG https_proxy

# Intel GPU driver versions (sync with .devops/openvino.Dockerfile)
ARG IGC_VERSION=v2.36.3
ARG IGC_VERSION_FULL=2_2.36.3+21719
ARG COMPUTE_RUNTIME_VERSION=26.22.38646.4
ARG COMPUTE_RUNTIME_VERSION_FULL=26.22.38646.4-0
ARG IGDGMM_VERSION=22.10.0

# Intel NPU driver versions (sync with .devops/openvino.Dockerfile)
ARG NPU_DRIVER_VERSION=v1.33.0
ARG NPU_DRIVER_FULL=v1.33.0.20260529-26625960453
ARG LIBZE1_VERSION=1.27.0-1~24.04~ppa2

# Install minimal runtime dependencies
# Deliberately NOT installing OpenVINO toolkit or OpenVINO packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libstdc++6 \
    wget \
    ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

# Install GPU drivers
RUN set -eux; \
    cd /tmp; \
    for url in \
        https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-core-${IGC_VERSION_FULL}_amd64.deb \
        https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-opencl-${IGC_VERSION_FULL}_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${IGDGMM_VERSION}_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb ; do \
        wget -q -O "$(basename "$url")" "$url"; \
    done; \
    apt-get update; \
    apt-get install -y --no-install-recommends ./*.deb; \
    rm -f ./*.deb; \
    rm -rf /var/lib/apt/lists/*

# Install NPU drivers
RUN set -eux; \
    cd /tmp; \
    wget -q -O npu-driver.tar.gz https://github.com/intel/linux-npu-driver/releases/download/${NPU_DRIVER_VERSION}/linux-npu-driver-${NPU_DRIVER_FULL}-ubuntu2404.tar.gz; \
    wget -q -O libze1.deb https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260324T100000Z/pool/main/l/level-zero-loader/libze1_${LIBZE1_VERSION}_amd64.deb; \
    mkdir npu && cd npu && tar -xf ../npu-driver.tar.gz && cp ../libze1.deb .; \
    apt-get update; \
    apt-get install -y --no-install-recommends ./*.deb; \
    cd / && rm -rf /tmp/npu /tmp/npu-driver.tar.gz /tmp/libze1.deb; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the extracted tarball contents
COPY test-openvino-extracted/llama-test/ /app/

# Create a simple test script
RUN cat > /app/test.sh << 'EOFTEST'
#!/bin/bash
set -e

MODEL="${MODEL_FILE:-/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
PASS=0
FAIL=0
SKIP=0

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }
skip() { echo "  SKIP: $1"; SKIP=$((SKIP + 1)); }

echo "================================================="
echo "Testing self-contained OpenVINO bundle"
echo "================================================="
echo ""
echo "Environment check:"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<not set>}"
echo "  System OpenVINO: $(ls /opt/intel/openvino* 2>/dev/null || echo 'NOT INSTALLED')"
echo "  GPU device:      $(ls /dev/dri/render* 2>/dev/null || echo 'NOT FOUND')"
echo "  NPU device:      $(ls /dev/accel* 2>/dev/null || echo 'NOT FOUND')"
echo "  Model: $MODEL"
echo ""

echo "--- Test 1: Version check ---"
if ./llama-cli --version; then
    pass "llama-cli --version"
else
    fail "llama-cli --version"
    exit 1
fi
echo ""

echo "--- Test 2: ldd check (all deps should be resolved) ---"
if ldd ./llama-cli | grep -E "not found"; then
    fail "unresolved dependencies found"
    exit 1
else
    pass "all dependencies resolved"
fi
echo ""

echo "--- Test 3: Bundled OpenVINO libraries ---"
echo "  OpenVINO core:    $(ls -1 libopenvino.so* 2>/dev/null | wc -l) files"
echo "  Plugin files:     $(ls -1 libopenvino_*_plugin.so 2>/dev/null | wc -l) files"
echo "  Compiler files:   $(ls -1 libopenvino_*_compiler*.so 2>/dev/null | wc -l) files"
echo "  TBB files:        $(ls -1 libtbb*.so* 2>/dev/null | wc -l) files"
echo ""

if [ ! -f "$MODEL" ]; then
    echo "WARNING: Model not found at $MODEL - skipping inference tests"
    skip "CPU inference (no model)"
    skip "GPU inference (no model)"
    skip "NPU inference (no model)"
else
    run_inference_test() {
        local device="$1"
        local log
        log=$(mktemp)
        printf 'What is 2+2?\nWhat is the capital of France?\n/exit\n' | \
            timeout 60 env GGML_OPENVINO_DEVICE="$device" ./llama-cli \
                -m "$MODEL" -c 64 --simple-io 2>&1 | tee "$log"
        local rc=${PIPESTATUS[1]}
        if [ $rc -eq 0 ] && grep -qi "4" "$log" && grep -qi "paris" "$log"; then
            pass "$device inference"
        else
            fail "$device inference (exit=$rc)"
        fi
        rm -f "$log"
    }

    echo "--- Test 4: CPU inference ---"
    run_inference_test CPU
    echo ""

    echo "--- Test 5: GPU inference ---"
    if [ -e /dev/dri/renderD128 ]; then
        run_inference_test GPU
    else
        skip "GPU inference (no /dev/dri/renderD128)"
    fi
    echo ""

    echo "--- Test 6: NPU inference ---"
    if [ -e /dev/accel/accel0 ]; then
        run_inference_test NPU
    else
        skip "NPU inference (no /dev/accel/accel0)"
    fi
fi
echo ""

echo "================================================="
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "================================================="

[ "$FAIL" -eq 0 ]
EOFTEST

RUN chmod +x /app/test.sh

# Volume mount point for model
VOLUME ["/models"]

CMD ["/app/test.sh"]
