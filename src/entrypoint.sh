#!/bin/sh
set -e

if [ -x "$(command -v nvidia-smi)" ]; then
    export CUDA_AVAILABLE=true
else
    export CUDA_AVAILABLE=false
fi

exec uvicorn app:app --host '0.0.0.0' --port '9528'
