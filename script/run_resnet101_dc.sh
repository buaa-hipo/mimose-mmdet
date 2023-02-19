#!/bin/bash

set -x

T=$(date '+%Y-%m-%d-%H-%M-%S')
if [ -z "${1+x}" ]; then
	for memory in 13 14 15 16 17; do
		python3 tools/train.py configs/retinanet/retinanet_r101_fpn_1x_coco_dc${memory}.py 2>&1 | tee log/resnet101_dc${memory}.log.$T
		sleep 300
	done
else
	memory=$1
	python3 tools/train.py configs/retinanet/retinanet_r101_fpn_1x_coco_dc${memory}.py 2>&1 | tee log/resnet101_dc${memory}.log.$T
fi
