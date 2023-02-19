T=`date '+%Y-%m-%d-%H-%M-%S'`

for memory in 6
do
    python3 tools/train.py configs/retinanet/retinanet_swin_small_fpn_1x_coco_sc${memory}.py 2>&1 | tee log/swin_small_sc${memory}.log.$T
    sleep 300
done
