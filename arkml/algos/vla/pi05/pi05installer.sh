#!/bin/bash
set -e

echo "=== ArkML Pi0.5 Dependency Installer ==="

echo "[1/6] Checking Python version..."
python3 --version

echo "[2/6] Installing base HF dependencies..."
pip install -U transformers accelerate huggingface_hub einops timm

echo "[3/6] Installing vision + dataset deps..."
pip install -U opencv-python pillow datasets decord

echo "[4/6] Installing LeRobot (main attempt)..."
pip install -U "git+https://github.com/huggingface/lerobot.git" || {
    echo "[WARN] Main LeRobot installation failed. Trying minimal fallback..."
    pip install lerobot || {
        echo "[ERROR] LeRobot installation failed."
        exit 1
    }
}

echo "[5/6] Logging in to HuggingFace CLI if needed..."
if ! huggingface-cli whoami >/dev/null 2>&1 ; then
    echo "HuggingFace authentication required. Please run:"
    echo "huggingface-cli login"
fi

echo "[6/6] Verifying PI05Policy import..."
python3 - << 'EOF'
from lerobot.policies.pi05 import PI05Policy
print("✓ PI05Policy imported successfully!")
# Weight loading skipped during install to avoid OOM

echo "=== Installation Complete ==="
