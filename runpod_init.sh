#!/usr/bin/env bash

apt-get update && apt-get install -y rsync
git pull
pip install -e '.[dev]'
pip install flash-attn --no-build-isolation
clx config --autoload-env on
export CLX_HOME=/workspace/clx
export HF_HOME=/workspace/hf
