T=`date '+%Y-%m-%d-%H-%M-%S'`

for memory in 6 9 10
do
    python3 tools/train.py configs/retinanet/retinanet_swin_small_fpn_1x_coco_dc${memory}.py 2>&1 | tee log/swin_small_dc${memory}.log.$T
    sleep 300
done
