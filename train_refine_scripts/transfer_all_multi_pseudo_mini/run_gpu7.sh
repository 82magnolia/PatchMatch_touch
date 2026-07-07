#!/usr/bin/env bash
bash "$(dirname "${BASH_SOURCE[0]}")/_run.sh" "${1:?Usage: run_gpu7.sh <GPU_ID>}"
