#!/bin/bash
# Script to Install OpenVINO Runtime on Linux from an Archive File
# https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-archive-linux.html#install-openvino-runtime-on-linux-from-an-archive-file

# === CONFIGURATION ===
export OPENVINO_VERSION_MAJOR=2025.3
export OPENVINO_VERSION_FULL=2025.3.0.19807.44526285f24
export OPENVINO_INSTALL_DIR=/opt/intel/openvino_${OPENVINO_VERSION_MAJOR}
export OPENVINO_LINK_DIR=/opt/intel/openvino

# === SCRIPT START ===

echo "🚀 Installing required system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libcurl4-openssl-dev \
    libtbb12 \
    cmake \
    ninja-build \
    python3-pip \
    curl \
    wget \
    tar \
    unzip

echo "📁 Creating installation directory..."
sudo mkdir -p /opt/intel

# if openvino_${OPENVINO_VERSION_MAJOR}.tgz folder already exists, then skip download and extraction
if [ -d "$OPENVINO_INSTALL_DIR" ]; then
    echo "⚠️ OpenVINO ${OPENVINO_VERSION_MAJOR} already installed at $OPENVINO_INSTALL_DIR. Skipping download and extraction."
else
    # --- DOWNLOAD ---
    DOWNLOAD_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION_MAJOR}/linux/openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64.tgz"
    TARBALL="openvino_${OPENVINO_VERSION_MAJOR}.tgz"

    echo "⬇️  Downloading OpenVINO ${OPENVINO_VERSION_FULL}..."
    if ! wget -O "$TARBALL" "$DOWNLOAD_URL"; then
        echo "❌ Failed to download from:"
        echo "   $DOWNLOAD_URL"
        exit 1
    fi

    # --- EXTRACT ---
    echo "📦 Extracting archive..."
    if ! tar -xf "$TARBALL"; then
        echo "❌ Failed to extract archive."
        rm -f "$TARBALL"
        exit 1
    fi

    EXTRACTED_DIR="openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64"
    if [ ! -d "$EXTRACTED_DIR" ]; then
        echo "❌ Extracted directory not found: $EXTRACTED_DIR"
        rm -f "$TARBALL"
        exit 1
    fi

    echo "🚚 Moving extracted files..."
    sudo mv "$EXTRACTED_DIR" "$OPENVINO_INSTALL_DIR"
    rm -f "$TARBALL"
fi

# --- INSTALL DEPENDENCIES ---
# Detect OS version
OS_FULL=$( . /etc/os-release ; echo "${ID}${VERSION_ID}" )

echo "Detected OS: ${OS_FULL}"

# Download custom dependency installer only for Ubuntu 25.04
if [ "$OS_FULL" == "ubuntu25.04" ]; then
    echo "Downloading custom dependency installer for Ubuntu 25.04..."
    CUSTOM_INSTALLER_URL="https://raw.githubusercontent.com/ravi9/misc-scripts/main/openvino/ov-archive-install/install_openvino_dependencies.sh"
    ORIGINAL_INSTALLER="$OPENVINO_INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh"

    if ! sudo wget -O "$ORIGINAL_INSTALLER" "$CUSTOM_INSTALLER_URL"; then
        echo "❌ Failed to download custom dependency installer from:"
        echo "   $CUSTOM_INSTALLER_URL"
        exit 1
    fi

    sudo chmod +x "$ORIGINAL_INSTALLER"
    echo "ℹ️  Custom dependency installer downloaded successfully for ${OS_FULL}."
fi

echo "🔧 Running OpenVINO dependency installer..."
cd "$OPENVINO_INSTALL_DIR"
if ! echo "Y" | sudo -E ./install_dependencies/install_openvino_dependencies.sh; then
    echo "❌ Dependency installation failed."
    exit 1
fi
cd -

# --- CREATE/UPDATE SYMLINK! ---
echo "🔗 Updating symlink to point to OpenVINO ${OPENVINO_VERSION_MAJOR}..."
sudo rm -f "$OPENVINO_LINK_DIR"  # Remove old symlink if exists
sudo ln -s "$OPENVINO_INSTALL_DIR" "$OPENVINO_LINK_DIR"

# --- SETUP ENVIRONMENT ---
echo "🎯 Setting up environment variables..."
source "$OPENVINO_LINK_DIR/setupvars.sh"

# --- VERIFY INSTALLATION ---
echo "✅ Verifying installation..."
if [ -f "$OPENVINO_INSTALL_DIR/runtime/version.txt" ]; then
    echo "✔️ OpenVINO installed successfully!"
    cat "$OPENVINO_INSTALL_DIR/runtime/version.txt"
else
    echo "⚠️ Could not verify version. ${OPENVINO_INSTALL_DIR}/runtime/version.txt. Installation may be incomplete."
fi

# --- FINAL MESSAGE ---
echo "🎉 All done! OpenVINO ${OPENVINO_VERSION_MAJOR} is now active."
echo "To use it in new terminals, run: source /opt/intel/openvino/setupvars.sh"
