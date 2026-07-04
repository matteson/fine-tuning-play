#!/usr/bin/env bash
set -euo pipefail  # -u catches unset vars, pipefail catches curl|bash failures

echo "========================================="
echo "Step 1: Install pixi"
echo "========================================="
# Correct URL — https://pixi.sh alone serves the homepage HTML, not the installer.
curl -fsSL https://pixi.sh/install.sh | bash
# The installer already appends the PATH line to ~/.bashrc; we only need it live
# in *this* non-interactive shell:
export PATH="$HOME/.pixi/bin:$PATH"

echo "========================================="
echo "Step 2: Init workspace"
echo "========================================="
rm -rf gemma-finetuning
pixi init gemma-finetuning
cd gemma-finetuning

echo "========================================="
echo "Step 3: Declare CUDA + point pypi at cu12x torch wheels"
echo "========================================="
# Tell pixi the target env has a CUDA 12 GPU so it won't resolve CPU-only variants.
# extra-index-urls gives uv access to the pytorch wheel index if you ever pin an
# exact +cuXXX build; the default PyPI `torch` linux wheel is already a CUDA build,
# so this is belt-and-suspenders, not required.
cat >> pixi.toml <<'EOF'

[system-requirements]
cuda = "12"

[pypi-options]
extra-index-urls = ["https://download.pytorch.org/whl/cu121"]
EOF

echo "========================================="
echo "Step 4: Conda layer = interpreter + kernel ONLY"
echo "========================================="
# Do NOT install pytorch/pytorch-cuda from conda — that's the ABI split that breaks
# unsloth/bitsandbytes/xformers. Keep conda to just the interpreter and jupyter glue.
pixi add python=3.10 ipykernel

echo "========================================="
echo "Step 5: Torch stack from PyPI (single ABI source)"
echo "========================================="
pixi add --pypi torch torchvision torchaudio

echo "========================================="
echo "Step 6: Training layer from PyPI"
echo "========================================="
# NOTE: unsloth pins transformers/trl tightly and will constrain (or reject) these
# upper bounds. If the solve fails here, install unsloth FIRST (Step 7) and let it
# choose transformers/trl, then relax these pins to match `pixi list`.
pixi add --pypi "transformers<=5.5.0" "datasets>=3.4.1,<4.4.0" "trl<=0.24.0" \
                peft accelerate bitsandbytes diffusers pydantic

echo "========================================="
echo "Step 7: Unsloth (auto-detects CUDA/arch; let it pin xformers/triton to torch)"
echo "========================================="
# Plain `unsloth` — the [cu121-ampere] extra is stale and fights the resolver.
pixi add --pypi unsloth

echo "========================================="
echo "Step 8: Jupyter kernel"
echo "========================================="
pixi run python -m ipykernel install --user \
    --name gemma-4-env --display-name "Gemma 4 (Pixi)"

echo "========================================="
echo "Step 9: Verify"
echo "========================================="
pixi list
pixi run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pixi run python -c "import unsloth; print('unsloth ok')"

echo "========================================="
echo "DONE. Refresh Jupyter and select the 'Gemma 4 (Pixi)' kernel."
echo "========================================="