#!/usr/bin/env bash

# Usage:
# wget https://raw.githubusercontent.com/ravi9/misc-scripts/refs/heads/main/openvino/install-gpu-npu-drivers/update-intel-gpu-npu-drivers.sh
# bash update-intel-gpu-npu-drivers.sh

set -Eeuo pipefail

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

TEMP_DIR=""
trap 'cleanup' EXIT INT TERM

log_info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()    { echo -e "${BLUE}[STEP]${NC} $*"; }
log_debug()   { echo -e "${CYAN}[DEBUG]${NC} $*"; }

cleanup() {
    [[ -n "${TEMP_DIR}" && -d "${TEMP_DIR}" ]] && rm -rf "${TEMP_DIR}"
}

# ----------------------------------------------------------------------
# Semantic version comparison (prefix only, up to length of v2)
# returns 0 (true) if v1 >= v2, 1 (false) otherwise
# ----------------------------------------------------------------------
ver_ge() {
    local v1="$1" v2="$2"
    [[ "$v1" == "none" ]] && return 1
    IFS='.' read -ra a1 <<< "$v1"
    IFS='.' read -ra a2 <<< "$v2"
    local max=${#a2[@]}
    for ((i=0; i<max; i++)); do
        local n1=${a1[i]:-0}
        local n2=${a2[i]:-0}
        (( n1 > n2 )) && return 0
        (( n1 < n2 )) && return 1
    done
    return 0
}

# ----------------------------------------------------------------------
# Get installed version of a package (without epoch/debian suffix)
# ----------------------------------------------------------------------
get_installed_version() {
    local pkg="$1"
    local ver
    ver=$(dpkg-query -W -f='${Version}' "$pkg" 2>/dev/null | cut -d':' -f2 | sed 's/[~-].*//' || true)
    [[ -z "$ver" ]] && echo "none" || echo "$ver"
}

# ----------------------------------------------------------------------
# Fetch latest release assets from GitHub, return a newline-separated list
# of "url|version" for each asset matching a given pattern.
# For .deb files: version is extracted from the filename (after _).
# For tarballs: version is the release tag (passed separately).
# ----------------------------------------------------------------------
fetch_assets_with_versions() {
    local repo="$1" pattern="$2"
    local api_url="https://api.github.com/repos/${repo}/releases/latest"
    local response
    response=$(curl -sSfL --retry 3 --retry-delay 1 "$api_url" 2>/dev/null) || {
        log_error "Failed to fetch GitHub API for ${repo}"
        return 1
    }
    # Extract tag version (for NPU we'll use this directly)
    local tag
    tag=$(echo "$response" | grep -o '"tag_name": *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"/\1/')
    tag="${tag#v}"
    # Extract URLs
    local urls
    urls=$(echo "$response" | grep -o '"browser_download_url": *"[^"]*"' | sed 's/.*"\([^"]*\)"/\1/' || true)
    local result=""
    while IFS= read -r url; do
        if [[ "$url" =~ $pattern ]]; then
            local version="$tag"
            # For .deb files, try to extract a more specific version from filename
            if [[ "$url" =~ \.deb$ ]]; then
                local filename=$(basename "$url")
                if [[ "$filename" =~ _([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+) ]]; then
                    version="${BASH_REMATCH[1]}"
                elif [[ "$filename" =~ _([0-9]+\.[0-9]+\.[0-9]+) ]]; then
                    version="${BASH_REMATCH[1]}"
                fi
            fi
            result+="${url}|${version}"$'\n'
        fi
    done <<< "$urls"
    echo "$result"
}

# ----------------------------------------------------------------------
# GPU update check (per package, using asset versions)
# ----------------------------------------------------------------------
check_gpu_updates() {
    log_step "Checking Intel GPU driver versions..."
    local repo="intel/compute-runtime"
    local assets_with_versions
    assets_with_versions=$(fetch_assets_with_versions "$repo" "\.deb$") || return 1

    declare -A pkg_pattern=(
        ["intel-opencl-icd"]="intel-opencl-icd_.*\.deb"
        ["intel-ocloc"]="intel-ocloc_.*\.deb"
        ["libigdgmm12"]="libigdgmm12_.*\.deb"
        ["libze-intel-gpu1"]="libze-intel-gpu1_.*\.deb"
    )
    declare -gA GPU_CR_LATEST
    declare -gA GPU_CR_ASSET
    GPU_CR_UPDATE_NEEDED=false

    for pkg in "${!pkg_pattern[@]}"; do
        local pattern="${pkg_pattern[$pkg]}"
        local match_url=""
        local match_ver=""
        while IFS='|' read -r url ver; do
            if [[ "$url" =~ $pattern ]]; then
                match_url="$url"
                match_ver="$ver"
                break
            fi
        done <<< "$assets_with_versions"
        if [[ -z "$match_url" ]]; then
            log_error "No asset found for $pkg"
            return 1
        fi
        GPU_CR_ASSET["$pkg"]="$match_url"
        GPU_CR_LATEST["$pkg"]="$match_ver"
        local installed=$(get_installed_version "$pkg")
        if [[ "$installed" == "none" ]] || ! ver_ge "$installed" "$match_ver"; then
            GPU_CR_UPDATE_NEEDED=true
        fi
    done

    # IGC (separate repo)
    local igc_repo="intel/intel-graphics-compiler"
    local igc_assets
    igc_assets=$(fetch_assets_with_versions "$igc_repo" "\.deb$") || return 1
    declare -A igc_pattern=(
        ["intel-igc-core-2"]="intel-igc-core-2_.*\.deb"
        ["intel-igc-opencl-2"]="intel-igc-opencl-2_.*\.deb"
    )
    declare -gA IGC_LATEST
    declare -gA IGC_ASSET
    GPU_IGC_UPDATE_NEEDED=false

    for pkg in "${!igc_pattern[@]}"; do
        local pattern="${igc_pattern[$pkg]}"
        local match_url=""
        local match_ver=""
        while IFS='|' read -r url ver; do
            if [[ "$url" =~ $pattern ]]; then
                match_url="$url"
                match_ver="$ver"
                break
            fi
        done <<< "$igc_assets"
        if [[ -z "$match_url" ]]; then
            log_error "No asset found for $pkg"
            return 1
        fi
        IGC_ASSET["$pkg"]="$match_url"
        IGC_LATEST["$pkg"]="$match_ver"
        local installed=$(get_installed_version "$pkg")
        if [[ "$installed" == "none" ]] || ! ver_ge "$installed" "$match_ver"; then
            GPU_IGC_UPDATE_NEEDED=true
        fi
    done
}

# ----------------------------------------------------------------------
# NPU update check (using tarball asset and release tag version)
# ----------------------------------------------------------------------
check_npu_updates() {
    log_step "Checking Intel NPU driver versions..."
    local repo="intel/linux-npu-driver"
    local api_url="https://api.github.com/repos/${repo}/releases/latest"
    local response
    response=$(curl -sSfL --retry 3 --retry-delay 1 "$api_url" 2>/dev/null) || {
        log_error "Failed to fetch NPU release info"
        return 1
    }
    local tag
    tag=$(echo "$response" | grep -o '"tag_name": *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"/\1/')
    tag="${tag#v}"
    NPU_LATEST="$tag"

    local tarball_url
    tarball_url=$(echo "$response" | grep -o '"browser_download_url": *"[^"]*"' | sed 's/.*"\([^"]*\)"/\1/' | grep -E "ubuntu2404.*\.tar\.gz" | head -1)
    if [[ -z "$tarball_url" ]]; then
        log_error "No Ubuntu 24.04 tarball found for NPU (expected filename containing 'ubuntu2404.tar.gz')"
        return 1
    fi
    NPU_ASSET_URL="$tarball_url"

    local pkg="intel-level-zero-npu"
    local installed=$(get_installed_version "$pkg")
    if [[ "$installed" == "none" ]] || ! ver_ge "$installed" "$NPU_LATEST"; then
        NPU_UPDATE_NEEDED=true
    else
        NPU_UPDATE_NEEDED=false
    fi
}

# ----------------------------------------------------------------------
# Print compact version summary (like original)
# ----------------------------------------------------------------------
print_summary() {
    echo -e "\n${CYAN}========== Version Summary ==========${NC}"
    echo -e "${BLUE}GPU Compute Runtime (intel-opencl-icd, ocloc, libigdgmm12, libze-intel-gpu1):${NC}"
    local cr_cur=()
    for pkg in intel-opencl-icd intel-ocloc libigdgmm12 libze-intel-gpu1; do
        cr_cur+=("$(get_installed_version "$pkg")")
    done
    echo "  Current versions: ${cr_cur[*]}"
    echo "  Latest:           ${GPU_CR_LATEST[intel-opencl-icd]} (runtime version)"
    echo "  Update needed:    ${GPU_CR_UPDATE_NEEDED}"

    echo -e "\n${BLUE}Intel Graphics Compiler (igc-core-2, igc-opencl-2):${NC}"
    local igc_cur=()
    for pkg in intel-igc-core-2 intel-igc-opencl-2; do
        igc_cur+=("$(get_installed_version "$pkg")")
    done
    echo "  Current versions: ${igc_cur[*]}"
    echo "  Latest:           ${IGC_LATEST[intel-igc-core-2]} (IGC version)"
    echo "  Update needed:    ${GPU_IGC_UPDATE_NEEDED}"

    echo -e "\n${BLUE}NPU Driver (intel-level-zero-npu):${NC}"
    echo "  Current version:  $(get_installed_version intel-level-zero-npu)"
    echo "  Latest:           ${NPU_LATEST}"
    echo "  Update needed:    ${NPU_UPDATE_NEEDED}"
    echo -e "${CYAN}=======================================${NC}\n"
}

# ----------------------------------------------------------------------
# Install .deb files from URLs
# ----------------------------------------------------------------------
install_debs_from_urls() {
    local urls=("$@")
    local deb_files=()
    for url in "${urls[@]}"; do
        local filename=$(basename "$url")
        local dest="${TEMP_DIR}/${filename}"
        log_info "Downloading ${filename} ..."
        curl -sSfL --retry 3 --retry-delay 1 -o "$dest" "$url" || return 1
        deb_files+=("$dest")
    done
    [[ ${#deb_files[@]} -eq 0 ]] && return 0
    log_info "Installing ${#deb_files[@]} package(s) with dpkg..."
    sudo dpkg -i "${deb_files[@]}" || sudo apt-get install -f -y
}

# ----------------------------------------------------------------------
# Perform GPU update
# ----------------------------------------------------------------------
perform_gpu_update() {
    log_step "Updating Intel GPU drivers..."
    local urls=()
    if [[ "$GPU_CR_UPDATE_NEEDED" == "true" ]]; then
        for pkg in intel-opencl-icd intel-ocloc libigdgmm12 libze-intel-gpu1; do
            urls+=("${GPU_CR_ASSET[$pkg]}")
        done
        install_debs_from_urls "${urls[@]}" || return 1
    fi
    if [[ "$GPU_IGC_UPDATE_NEEDED" == "true" ]]; then
        urls=()
        for pkg in intel-igc-core-2 intel-igc-opencl-2; do
            urls+=("${IGC_ASSET[$pkg]}")
        done
        install_debs_from_urls "${urls[@]}" || return 1
    fi
}

# ----------------------------------------------------------------------
# Perform NPU update
# ----------------------------------------------------------------------
perform_npu_update() {
    log_step "Updating Intel NPU driver..."
    [[ "$NPU_UPDATE_NEEDED" != "true" ]] && { log_info "NPU driver already up to date."; return 0; }
    log_info "Downloading NPU tarball (${NPU_LATEST}) ..."
    local tarball="${TEMP_DIR}/npu_driver.tar.gz"
    curl -sSfL --retry 3 --retry-delay 1 -o "$tarball" "${NPU_ASSET_URL}" || return 1
    local extract_dir="${TEMP_DIR}/npu_extract"
    mkdir -p "$extract_dir"
    tar -xzf "$tarball" -C "$extract_dir" || return 1
    mapfile -t debs < <(find "$extract_dir" -name '*.deb' -type f)
    [[ ${#debs[@]} -eq 0 ]] && { log_error "No .deb files in NPU tarball"; return 1; }
    sudo dpkg --purge --force-remove-reinstreq intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu 2>/dev/null || true
    sudo dpkg -i "${debs[@]}" || sudo apt-get install -f -y
    if ! dpkg-query -W level-zero-loader >/dev/null 2>&1; then
        sudo apt-get install -y level-zero-loader
    fi
}

# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
validate_installation() {
    log_step "Validating driver installation..."
    echo -e "${BLUE}--- Installed Intel packages ---${NC}"
    dpkg -l | grep -i intel | grep -E 'igc|opencl|ocloc|igdgmm|level-zero|gpu' || echo "None found."
    echo -e "${BLUE}--- OpenCL platforms ---${NC}"
    command -v clinfo &>/dev/null && clinfo -l || echo "clinfo not installed."
    echo -e "${BLUE}--- GPU render nodes ---${NC}"
    ls -l /dev/dri/render* 2>/dev/null || echo "None."
    echo -e "${BLUE}--- NPU acceleration nodes ---${NC}"
    ls -l /dev/accel/accel* 2>/dev/null || echo "None."
    echo -e "${BLUE}--- Group memberships (render, video) ---${NC}"
    groups | grep -E "render|video" || echo "User not in render/video groups."
    if command -v python3 &>/dev/null && python3 -c "import openvino" &>/dev/null; then
        echo -e "${BLUE}--- OpenVINO available devices ---${NC}"
        python3 -c "import openvino as ov; print('Devices:', ov.Core().available_devices)"
    fi
}

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
main() {
    local AUTO_YES=false
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -y|--yes) AUTO_YES=true ;;
            *) log_error "Unknown option: $1"; exit 1 ;;
        esac
        shift
    done

    log_info "Intel GPU/NPU Driver Auto-Updater started."
    if ! grep -q "Ubuntu 24.04" /etc/os-release; then
        log_error "This script is designed for Ubuntu 24.04 only."
        exit 1
    fi

    TEMP_DIR=$(mktemp -d)
    log_debug "Using temporary directory: ${TEMP_DIR}"

    check_gpu_updates || exit 1
    check_npu_updates || exit 1
    print_summary

    if [[ "$GPU_CR_UPDATE_NEEDED" != "true" && "$GPU_IGC_UPDATE_NEEDED" != "true" && "$NPU_UPDATE_NEEDED" != "true" ]]; then
        log_info "All drivers are already up to date. Exiting."
        exit 0
    fi

    if [[ "$AUTO_YES" == false ]]; then
        read -p "Do you wish to update the drivers listed above? (y/N) " -n 1 -r
        echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && { log_info "Update cancelled."; exit 0; }
    fi

    sudo apt update -y
    sudo apt install -y curl wget dpkg clinfo 2>/dev/null || true

    for grp in render video; do
        if ! groups | grep -q "\b${grp}\b"; then
            sudo usermod -aG "${grp}" "$USER"
            log_warn "Added user to group '${grp}' – changes take effect next login."
        fi
    done

    perform_gpu_update || { log_error "GPU update failed"; exit 1; }
    perform_npu_update || { log_error "NPU update failed"; exit 1; }
    sudo apt-get install -f -y

    validate_installation
    log_info "All operations completed successfully."
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
