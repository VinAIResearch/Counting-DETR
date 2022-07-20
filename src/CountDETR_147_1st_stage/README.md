COPY from https://github.com/megvii-research/AnchorDETR

**Anchor DETR**: Query Design for Transformer-Based Detector
========


## Introduction
This repository is an official implementation of the [Anchor DETR](https://arxiv.org/abs/2109.07107).
We encode the anchor points as the object queries in DETR.
Multiple patterns are attached to each anchor point to solve the difficulty: "one region, multiple objects".
We also propose an attention variant RCDA to reduce the memory cost for high-resolution features.


![DETR](.github/pipeline.png)


## Main Results




*Note:*
1. The results are based on ResNet-50 backbone.
2. Inference speeds are measured on NVIDIA Tesla V100 GPU.
3. The values in parentheses of the Infer Speed indicate the speed with torchscript optimization.

### Training
To train AnchorDETR on a single node with 8 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py  --coco_path /path/to/coco 
```

### Evaluation
To evaluate AnchorDETR on a single node with 8 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --eval --coco_path /path/to/coco --resume /path/to/checkpoint.pth 
```

To evaluate AnchorDETR with a single GPU:
```
python main.py --eval --coco_path /path/to/coco --resume /path/to/checkpoint.pth
```