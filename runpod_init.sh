#!/usr/bin/env bash

git fetch origin
git reset --hard origin/main
pip install -e .
pip install flash-attn --no-build-isolation
clx config --autoload-env on
export CLX_HOME=/workspace/clx/home
export HF_HOME=/workspace/hf
