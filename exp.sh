#!/bin/bash

set -x

export PROJECT_ROOT=$(pwd)
export CUDA_VISIBLE_DEVICES=0

mkdir -p log

# OD-R50
bash script/run_resnet50_dc.sh 15

# OD-R101
bash script/run_resnet101_dc.sh 13
