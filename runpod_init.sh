#!/usr/bin/env bash

apt-get update && apt-get install -y rsync
git pull
pip install -e '.[dev]'
pip install flash-attn --no-build-isolation
clx config --autoload-env on
cat > .env << 'EOF'
CLX_HOME=/workspace/clx
HF_HOME=/workspace/hf
EOF
