#!/bin/bash

set -x

if [ -z "${1+x}" ]; then
	for memory in 15 16; do
		T=$(date '+%Y-%m-%d-%H-%M-%S')
		python3 tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco_dc${memory}.py 2>&1 | tee log/resnet50_dc${memory}.log.$T
		sleep 300
	done
else
	T=$(date '+%Y-%m-%d-%H-%M-%S')
	memory=$1
	python3 tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco_dc${memory}.py 2>&1 | tee log/resnet50_dc${memory}.log.$T
fi
