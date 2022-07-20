**Count DETR**: Few-shot Object Counting and Detection
========


## Introduction
This repository is an non-official, temporary  implementation of the Count DETR.
We use two stages training to generate pseudo label to alleviate the requirment for large data.
Uncertainty module is used to mitigate the imperfection of pseudo label.

![DETR](images/MainArch.png)


## Main Results



|                    | Dataset       |  MAE    |  AP     |  GFLOPs  | Infer Speed (FPS) |
|:------------------:|:-------------:|:-----:|:-------:|:--------:|:-----------------:|
| Count DETR         |  FSCD 147     |  500    |  43.3   |  187     | 10 (12)           |
| Count DETR         |  FSCD LVIS    |  50     |  43.7   |  152     | 10                |


*Note:*
1. The results are based on ResNet-50 backbone.
2. Inference speeds are measured on NVIDIA Tesla V100 GPU.
3. The values in parentheses of the Infer Speed indicate the speed with torchscript optimization.


## Usage

### Installation
First, clone the repository locally:
```
git clone https://github.com/megvii-research/AnchorDETR.git
```
Then, install dependencies:
```
pip install -r requirements.txt
```

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


## Citation

If you find this project useful for your research, please consider citing the paper.
```
@misc{wang2021anchor,
      title={Anchor DETR: Query Design for Transformer-Based Detector},
      author={Yingming Wang and Xiangyu Zhang and Tong Yang and Jian Sun},
      year={2021},
      eprint={2109.07107},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact
If you have any questions, feel free to open an issue or contact us at wangyingming@megvii.com.
