#!/bin/bash
# ============================================================================
# download_data.sh — 下载 Curated Nerfbusters 数据集
#
# 尝试多种方式下载：
#   1. gdown (标准方式)
#   2. wget + proxy (需要配置 https_proxy)
#   3. 手动提示
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

DATASET_DIR="data"
mkdir -p "${DATASET_DIR}"

echo "=========================================="
echo "  Downloading Curated Nerfbusters Dataset"
echo "=========================================="

# 文件信息
FILE_ID="1lTtSy578gYkoVVsFiB3PosLK9JOOYhbC"
FILE_NAME="curated-nerfbusters.zip"
TARGET_DIR="${DATASET_DIR}/curated_nb"

# 方法 1: gdown
download_gdown() {
    echo "[Method 1] Trying gdown..."
    pip install gdown -q 2>/dev/null

    if command -v gdown &>/dev/null; then
        gdown "${FILE_ID}" -O "/tmp/${FILE_NAME}" && return 0
    fi

    python -c "
import gdown
try:
    gdown.download(id='${FILE_ID}', output='/tmp/${FILE_NAME}', quiet=False)
    import os
    if os.path.getsize('/tmp/${FILE_NAME}') > 1000000:
        exit(0)
    exit(1)
except:
    exit(1)
" && return 0

    return 1
}

# 方法 2: 通过代理使用 gdown
download_gdown_proxy() {
    echo "[Method 2] Trying gdown with proxy..."
    if [ -n "${https_proxy}" ] || [ -n "${HTTPS_PROXY}" ]; then
        python -c "
import os
import gdown
# Try with SSL verification disabled
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass
try:
    gdown.download(id='${FILE_ID}', output='/tmp/${FILE_NAME}', quiet=False)
    import os
    if os.path.getsize('/tmp/${FILE_NAME}') > 1000000:
        exit(0)
    exit(1)
except:
    exit(1)
" && return 0
    fi
    return 1
}

# 方法 3: 使用 curl + cookie
download_curl() {
    echo "[Method 3] Trying curl with Google Drive cookie..."

    # 获取确认页面
    curl -L -c /tmp/gdrive_cookies.txt \
        "https://docs.google.com/uc?export=download&id=${FILE_ID}" \
        -o /tmp/gdrive_page.html 2>/dev/null

    # 提取确认码
    CONFIRM=$(grep -o "confirm=[^&\"]*" /tmp/gdrive_page.html | head -1 | sed 's/confirm=//')

    if [ -n "${CONFIRM}" ]; then
        curl -L -b /tmp/gdrive_cookies.txt \
            "https://docs.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
            -o "/tmp/${FILE_NAME}" 2>/dev/null

        if [ -f "/tmp/${FILE_NAME}" ] && [ "$(stat -c%s "/tmp/${FILE_NAME}")" -gt 1000000 ]; then
            return 0
        fi
    fi
    return 1
}

# 方法 4: 手动提示
manual_download() {
    echo ""
    echo "=========================================="
    echo "  Manual Download Required"
    echo "=========================================="
    echo ""
    echo "Automated download failed. Please manually:"
    echo ""
    echo "1. Open this link in your browser:"
    echo "   https://drive.google.com/file/d/${FILE_ID}/view"
    echo ""
    echo "2. Download ${FILE_NAME}"
    echo ""
    echo "3. Place it at: ${REPO_DIR}/${DATASET_DIR}/${FILE_NAME}"
    echo ""
    echo "4. Then run:"
    echo "   unzip ${REPO_DIR}/${DATASET_DIR}/${FILE_NAME} -d ${REPO_DIR}/${DATASET_DIR}/"
    echo ""
    echo "=========================================="
}

# ---- 主流程 ----
# 解压已有文件
if [ -f "${DATASET_DIR}/${FILE_NAME}" ] && [ "$(stat -c%s "${DATASET_DIR}/${FILE_NAME}")" -gt 1000000 ]; then
    echo "Dataset zip already exists. Extracting..."
    unzip -o "${DATASET_DIR}/${FILE_NAME}" -d "${TARGET_DIR}"
    echo "Extracted to ${TARGET_DIR}/"
    exit 0
fi

# 尝试下载
download_gdown || download_gdown_proxy || download_curl || {
    manual_download
    exit 1
}

# 验证 + 解压
if [ -f "/tmp/${FILE_NAME}" ] && [ "$(stat -c%s "/tmp/${FILE_NAME}")" -gt 1000000 ]; then
    echo "Download successful! Extracting..."
    mv "/tmp/${FILE_NAME}" "${DATASET_DIR}/${FILE_NAME}"
    unzip -o "${DATASET_DIR}/${FILE_NAME}" -d "${TARGET_DIR}"
    echo ""
    echo "Dataset ready at: ${TARGET_DIR}/"
    echo ""
    echo "Available scenes:"
    ls "${TARGET_DIR}/" 2>/dev/null
else
    echo "Download failed (file too small or missing)."
    manual_download
    exit 1
fi

echo "Done!"
