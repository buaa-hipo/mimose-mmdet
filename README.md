## Requirements

- V100
- Docker with functional NVIDIA GPU support

## Install

1. Create a docker container with NVIDIA GPU enabled (`--shm-size` must be set large enough for PyTorch dataloader workers)

   ```bash
   docker run --name mimose -itd --gpus all --shm-size 32G -v <dataset_path>:/opt/dataset pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel bash
   docker exec -it mimose bash
   ```

2. Install Git using `apt`

   ```bash
   chmod 777 /tmp # apt update would fail without this
   apt update
   apt install -y git
   ```

3. Setup conda, create a new env and install PyTorch

   ```bash
   # Setup conda
   conda init
   . ~/.bashrc
   
   # Create conda env and install PyTorch
   conda create -n mimose python=3.9
   conda activate mimose
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
   ```
   
4. Install `mimose-mmdet`  and dependencies (download coco dataset if not exist)

   ```bash
   # Setup mimose-mmdet repo and install dependencies
   git clone https://github.com/mimose-project/mimose-mmdet && cd mimose-mmdet
   pip install cython mmcv-full
   apt install libgl1 libglib2.0-0 # required by opencv
   pip install -v -e .
   
   # Create dataset symlink
   ln -s /opt/dataset ./data # assume coco dataset is located at `/opt/dataset/coco`
   ```

## Getting Started

1. Run the evaluation scripts for mimose:

   ```bash
   cd mimose-mmdet
   # Run the evaluation all-in-one script!
   bash exp.sh
   ```

2. Check logs in `./log` directory

3. You can also run seperate evaluation scripts executed in `exp.sh` manually.

